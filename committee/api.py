from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from committee.config import settings
from committee.models import TraceEvent, Thesis

app = FastAPI(title="Investment Committee")

_runs: dict[str, dict] = {}
_TERMINAL_EVENT = "debate_finished"


class DebateRequest(BaseModel):
    thesis: str
    ticker: str | None = None
    entity: str | None = None
    budget: int = settings.default_budget
    policy: str = "explore_exploit"
    offline: bool = False


async def _execute(run_id: str, req: DebateRequest) -> None:
    from committee.debate.graph import build_graph, build_runtime, build_trace

    entry = _runs[run_id]
    try:
        rt, writer = build_runtime(
            thesis=Thesis(statement=req.thesis, ticker=req.ticker, entity=req.entity),
            budget=req.budget, policy=req.policy, offline=req.offline,
            run_dir=Path(settings.runs_dir), run_id=run_id,
        )
    except Exception as exc:
        entry["status"] = f"failed: {exc}"
        return
    entry["status"] = "running"

    def fanout(event: TraceEvent) -> None:
        for q in entry["queues"]:
            q.put_nowait(event)

    rt.bus.subscribe(fanout)
    await rt.bus.publish("debate_started", thesis=req.thesis, budget=req.budget, policy=req.policy)
    try:
        graph = build_graph()
        await graph.ainvoke({"rt": rt}, config={"recursion_limit": settings.max_rounds * 4 + 8})
        entry["status"] = "done"
    except Exception as exc:
        entry["status"] = f"failed: {exc}"
    finally:
        await rt.bus.publish(_TERMINAL_EVENT, run_id=run_id)
        trace = build_trace(rt)
        entry["trace"] = trace
        writer.finalize(trace)


@app.post("/debate")
async def start_debate(req: DebateRequest) -> dict:
    import uuid

    run_id = uuid.uuid4().hex[:10]
    _runs[run_id] = {"status": "starting", "trace": None, "queues": [], "rt": None}
    asyncio.create_task(_execute(run_id, req))
    return {"run_id": run_id, "stream": f"/debate/{run_id}/stream"}


@app.get("/debate/{run_id}")
async def get_debate(run_id: str) -> dict:
    entry = _runs.get(run_id)
    if not entry:
        raise HTTPException(404, "unknown run")
    trace = entry["trace"]
    return {"status": entry["status"], "trace": json.loads(trace.model_dump_json()) if trace else None}


@app.get("/debate/{run_id}/stream")
async def stream_debate(run_id: str) -> StreamingResponse:
    entry = _runs.get(run_id)
    if not entry:
        raise HTTPException(404, "unknown run")

    queue: asyncio.Queue[TraceEvent] = asyncio.Queue()
    entry["queues"].append(queue)

    async def generate():
        while True:
            event = await queue.get()
            yield f"event: {event.type}\ndata: {event.model_dump_json()}\n\n"
            if event.type == _TERMINAL_EVENT:
                break

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/runs")
async def list_runs() -> list[dict]:
    return [{"run_id": rid, "status": e["status"]} for rid, e in _runs.items()]
