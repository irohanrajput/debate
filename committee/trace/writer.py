from __future__ import annotations

from pathlib import Path

from committee.models import CommitteeMemo, DebateTrace, TraceEvent


class TraceWriter:
    """Bus subscriber: streams events.jsonl live; writes trace.json + memo.md at end."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        self._events_file = (run_dir / "events.jsonl").open("a")

    def __call__(self, event: TraceEvent) -> None:
        if self._events_file.closed:
            return
        self._events_file.write(event.model_dump_json() + "\n")
        self._events_file.flush()

    def finalize(self, trace: DebateTrace) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "trace.json").write_text(trace.model_dump_json(indent=2))
        if trace.memo:
            (self.run_dir / "memo.md").write_text(render_memo(trace.memo))
        self._events_file.close()


def render_memo(memo: CommitteeMemo) -> str:
    lines = [
        f"# Committee memo: {memo.recommendation.value}",
        "",
        f"**{memo.headline}**",
        "",
        f"Confidence: {memo.confidence:.2f}" + (f" | Position guidance: {memo.position_guidance}" if memo.position_guidance else ""),
        "",
        "## Analyst verdicts",
        "",
    ]
    lines += [f"- **{v.lens}** — {v.stance.value} ({v.confidence:.2f}): {v.summary}" for v in memo.verdicts]
    if memo.resolved:
        lines += ["", "## Resolved disagreements", ""]
        lines += [f"- {r.claim_id} [{r.method.value}] {r.verdict}: {r.reasoning}" for r in memo.resolved]
    if memo.unresolved:
        lines += ["", "## Unresolved disagreements", ""]
        lines += [f"- {r.claim_id}: {r.reasoning}" for r in memo.unresolved]
    if memo.reasoning_trace:
        lines += ["", "## Reasoning trace", ""]
        lines += [f"- R{t.round}/{t.lens}" + (f"/{t.claim_id}" if t.claim_id else "") + f": {t.note}" for t in memo.reasoning_trace]
    lines += ["", f"_Data as of: {memo.data_as_of}_" if memo.data_as_of else ""]
    return "\n".join(lines)
