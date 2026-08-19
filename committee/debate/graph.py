from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from committee.config import settings
from committee.agents.analyst import Analyst, render_findings, render_others
from committee.agents.chair import synthesize
from committee.agents.lenses import LENSES
from committee.debate.budget import InsufficientBudget, Ledger, Pool
from committee.debate.disagreement import assess, must_address_for
from committee.debate.policies.base import get_policy
from committee.debate.resolver import resolve_conflicts
from committee.debate.state import DebateRuntime, GraphState
from committee.evidence.corpus import CorpusIndex, NullCorpus
from committee.evidence.market import MarketSnapshot
from committee.evidence.store import EvidenceStore
from committee.evidence.tools import ToolBox
from committee.llm.client import GeminiProvider
from committee.llm.embeddings import get_embedder
from committee.models import (
    AnalystMemory,
    DebateState,
    DebateTrace,
    Evidence,
    Findings,
    Mode,
    RoundRecord,
    Thesis,
    Tier,
)
from committee.trace.events import EventBus
from committee.trace.writer import TraceWriter


async def _allocate(state: GraphState) -> GraphState:
    rt = state["rt"]
    rt.round += 1
    rt.decision = rt.policy.allocate(rt.state, rt.ledger, rt.round, rt.lenses)
    await rt.bus.publish("budget_decision", round=rt.round, mode=rt.decision.mode.value,
                         selected=rt.decision.selected_lenses, rationale=rt.decision.rationale,
                         research=rt.decision.research_by_lens, argue=rt.decision.argue_by_lens)
    if not rt.rounds or rt.rounds[-1].mode != rt.decision.mode:
        await rt.bus.publish("mode_changed", round=rt.round, mode=rt.decision.mode.value)
    return {"rt": rt}


async def _run_one(rt: DebateRuntime, lens: str) -> tuple[str, object | None]:
    decision, analyst, memory = rt.decision, rt.analysts[lens], rt.memories[lens]
    tier = decision.tier_by_lens.get(lens, Tier.FLASH)
    evidence: list[Evidence] = [ev for eid in memory.evidence_ids if (ev := rt.store.get(eid))]

    research_alloc = decision.research_by_lens.get(lens, 0)
    if research_alloc > 0:
        res = rt.ledger.reserve(Pool.RESEARCH, lens, "plan", min(research_alloc, settings.plan_tokens))
        if res.tokens > 0:
            await rt.bus.publish("agent_started", round=rt.round, lens=lens, phase="research")
            try:
                focus_parts = []
                if decision.mode == Mode.EXPLOIT:
                    focus_parts.append("Focus ONLY on the contested claims you must address.")
                if not rt.thesis.ticker:
                    focus_parts.append("This entity has no public market data; do not call market tools; rely on search_corpus.")
                focus = " ".join(focus_parts)
                new_evidence, usage = await analyst.research(rt.thesis, tier, res.tokens, focus, rt.round)
                rt.ledger.commit(res, usage)
                for ev in new_evidence:
                    if ev.id not in memory.evidence_ids:
                        memory.evidence_ids.append(ev.id)
                        evidence.append(ev)
            except InsufficientBudget as exc:
                rt.ledger.release(res)
                await rt.bus.publish("agent_skipped", round=rt.round, lens=lens, reason=f"research: {exc}")
            except Exception as exc:
                rt.ledger.release(res)
                await rt.bus.publish("error", round=rt.round, lens=lens, where="research", detail=str(exc))
        else:
            rt.ledger.release(res)

    argue_alloc = decision.argue_by_lens.get(lens, 0)
    res = rt.ledger.reserve(Pool.DEBATE, lens, "argue", argue_alloc)
    if res.tokens < settings.argue_floor_tokens:
        rt.ledger.release(res)
        await rt.bus.publish("agent_skipped", round=rt.round, lens=lens, reason="budget below floor")
        return lens, None
    await rt.bus.publish("agent_started", round=rt.round, lens=lens, phase="argue")
    try:
        if rt.round == 1:
            output = await analyst.discover(rt.thesis, evidence, tier, res.tokens, rt.round)
            memory.findings = output
        else:
            prior_round = rt.rounds[-1]
            others = (render_findings(prior_round.findings, lens) if prior_round.findings
                      else render_others(prior_round.positions, lens))
            must = must_address_for(lens, rt.state) if rt.state else []
            output = await analyst.argue(rt.thesis, evidence, others, must,
                                         rt.decision.mode, tier, res.tokens, rt.round)
            memory.positions.append(output)
        rt.ledger.commit(res, output.usage)
        await rt.bus.publish("agent_done", round=rt.round, lens=lens,
                             stance=getattr(output, "stance", None),
                             lean=str(getattr(output, "provisional_lean", "") or ""),
                             summary=output.summary)
        return lens, output
    except InsufficientBudget as exc:
        rt.ledger.release(res)
        await rt.bus.publish("agent_skipped", round=rt.round, lens=lens, reason=str(exc))
        return lens, None
    except Exception as exc:
        rt.ledger.release(res)
        await rt.bus.publish("error", round=rt.round, lens=lens, where="argue", detail=str(exc))
        return lens, None


async def _run_round(state: GraphState) -> GraphState:
    rt = state["rt"]
    results = await asyncio.gather(*(_run_one(rt, lens) for lens in rt.decision.selected_lenses))
    record = RoundRecord(round=rt.round, mode=rt.decision.mode, budget_decision=rt.decision)
    for lens, output in results:
        if output is None:
            continue
        if isinstance(output, Findings):
            record.findings[lens] = output
        else:
            record.positions[lens] = output
    rt.rounds.append(record)
    await rt.bus.publish("budget_update", round=rt.round, **rt.ledger.summary())
    return {"rt": rt}


async def _assess(state: GraphState) -> GraphState:
    rt = state["rt"]
    record = rt.rounds[-1]
    if record.findings and not record.positions:
        questions = [q for f in record.findings.values() for q in f.open_questions]
        rt.state = DebateState(round=rt.round, mode=Mode.DISCOVER, coverage_notes=questions)
        await rt.bus.publish("coverage", round=rt.round, open_questions=questions)
    else:
        positions = {lens: m.latest_position for lens, m in rt.memories.items() if m.latest_position}
        rt.state = await assess(round=rt.round, positions=positions, prior=rt.state,
                                embedder=rt.embedder, provider=rt.provider, ledger=rt.ledger,
                                judge_cache=rt.judge_cache)
        record.state = rt.state
        await rt.bus.publish("disagreement_update", round=rt.round,
                             score=rt.state.disagreement_score, delta=rt.state.convergence_delta,
                             contested=[c.model_dump() for c in rt.state.contested],
                             converged=rt.state.converged)
        for c in rt.state.contested:
            await rt.bus.publish("contested_claim", round=rt.round, claim_id=c.claim_id,
                                 owner=c.owner, against=c.against_lenses, score=c.score)
        if rt.state.converged:
            await rt.bus.publish("convergence", round=rt.round, score=rt.state.disagreement_score)
    return {"rt": rt}


def _should_continue(state: GraphState) -> str:
    rt = state["rt"]
    if rt.round >= settings.max_rounds:
        return "stop"
    if rt.round >= settings.min_rounds and rt.state and rt.state.converged:
        return "stop"
    if rt.ledger.remaining(Pool.DEBATE) < settings.argue_floor_tokens * len(rt.lenses):
        return "stop"
    return "continue"


async def _resolve(state: GraphState) -> GraphState:
    rt = state["rt"]
    contested = rt.state.contested if rt.state else []
    positions = {lens: m.latest_position for lens, m in rt.memories.items() if m.latest_position}
    if contested:
        rt.resolutions = await resolve_conflicts(contested=contested, positions=positions,
                                                 provider=rt.provider, ledger=rt.ledger)
        for r in rt.resolutions:
            await rt.bus.publish("resolution", claim_id=r.claim_id, method=r.method.value,
                                 verdict=r.verdict, reasoning=r.reasoning)
    return {"rt": rt}


async def _synthesize(state: GraphState) -> GraphState:
    rt = state["rt"]
    await rt.bus.publish("synthesis_started")
    finals = {lens: m.latest_position for lens, m in rt.memories.items() if m.latest_position}
    rt.memo = await synthesize(thesis=rt.thesis, rounds=rt.rounds, finals=finals,
                               resolutions=rt.resolutions, provider=rt.provider,
                               ledger=rt.ledger, data_as_of=rt.data_as_of)
    await rt.bus.publish("memo_ready", recommendation=rt.memo.recommendation.value,
                         headline=rt.memo.headline, confidence=rt.memo.confidence)
    return {"rt": rt}


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("allocate", _allocate)
    graph.add_node("run_round", _run_round)
    graph.add_node("assess", _assess)
    graph.add_node("resolve", _resolve)
    graph.add_node("synthesize", _synthesize)
    graph.add_edge(START, "allocate")
    graph.add_edge("allocate", "run_round")
    graph.add_edge("run_round", "assess")
    graph.add_conditional_edges("assess", _should_continue, {"continue": "allocate", "stop": "resolve"})
    graph.add_edge("resolve", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()


def build_runtime(*, thesis: Thesis, budget: int, policy: str, offline: bool,
                  run_dir: Path, provider=None, embedder=None, corpus=None,
                  snapshot: MarketSnapshot | None = None,
                  run_id: str | None = None) -> tuple[DebateRuntime, TraceWriter]:
    run_id = run_id or uuid.uuid4().hex[:10]
    bus = EventBus(run_id)
    writer = TraceWriter(run_dir / run_id)
    bus.subscribe(writer)
    ledger = Ledger(total=budget, on_event=None)

    snapshot = snapshot or MarketSnapshot()
    if offline:
        snapshot.offline = True
    else:
        snapshot.fetch_eager(thesis.ticker)
    if corpus is None:
        try:
            corpus = CorpusIndex()
        except Exception:
            corpus = NullCorpus()
    store = EvidenceStore()
    toolbox = ToolBox(snapshot, corpus, store)
    provider = provider or GeminiProvider()
    embedder = embedder or get_embedder()

    rt = DebateRuntime(
        run_id=run_id, thesis=thesis, lenses=list(LENSES), policy=get_policy(policy),
        provider=provider, embedder=embedder, toolbox=toolbox, store=store,
        ledger=ledger, bus=bus, data_as_of=snapshot.as_of,
    )
    for name, spec in LENSES.items():
        async def emit(event_type: str, payload: dict, _lens=name) -> None:
            await bus.publish(event_type, round=rt.round, lens=_lens, **payload)
        rt.analysts[name] = Analyst(spec, provider, toolbox, emit)
        rt.memories[name] = AnalystMemory(lens=name)
    return rt, writer


def build_trace(rt: DebateRuntime) -> DebateTrace:
    return DebateTrace(
        run_id=rt.run_id, thesis=rt.thesis,
        config_snapshot={"policy": rt.policy.name, "budget": rt.ledger.total,
                         "min_rounds": settings.min_rounds, "max_rounds": settings.max_rounds,
                         "lenses": rt.lenses},
        rounds=rt.rounds, resolutions=rt.resolutions, memo=rt.memo,
        totals={**rt.ledger.summary(), "evidence_count": len(rt.store.index()),
                "budget_history": rt.ledger.history},
    )


async def run_debate(*, thesis: Thesis, budget: int, policy: str = "explore_exploit",
                     offline: bool = False, run_dir: Path | None = None,
                     subscribers: list | None = None, provider=None, embedder=None,
                     corpus=None) -> DebateTrace:
    rt, writer = build_runtime(thesis=thesis, budget=budget, policy=policy, offline=offline,
                               run_dir=run_dir or Path(settings.runs_dir),
                               provider=provider, embedder=embedder, corpus=corpus)
    for sub in subscribers or []:
        rt.bus.subscribe(sub)
    await rt.bus.publish("debate_started", thesis=thesis.statement, budget=budget, policy=policy)
    graph = build_graph()
    try:
        await graph.ainvoke({"rt": rt}, config={"recursion_limit": settings.max_rounds * 4 + 8})
    finally:
        await rt.bus.publish("debate_finished", run_id=rt.run_id)
        trace = build_trace(rt)
        writer.finalize(trace)
    return trace
