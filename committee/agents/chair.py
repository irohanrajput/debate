from __future__ import annotations

from pathlib import Path

from committee.config import settings
from committee.debate.budget import Ledger, Pool, estimate_tokens
from committee.llm.client import LLMProvider
from committee.models import (
    AnalystPosition,
    AnalystVerdict,
    CommitteeMemo,
    Resolution,
    ResolutionMethod,
    RoundRecord,
    Thesis,
    Tier,
)

_PROMPTS = Path(__file__).parent / "prompts"


def _rounds_summary(rounds: list[RoundRecord]) -> str:
    lines = []
    for r in rounds:
        if r.findings:
            obs = " | ".join(f"{lens}: {f.summary}" for lens, f in r.findings.items())
            lines.append(f"R{r.round} ({r.mode.value}): {obs}")
        if r.positions:
            pos = " | ".join(f"{lens}: {p.stance.value}({p.confidence:.2f})" for lens, p in r.positions.items())
            score = r.state.disagreement_score if r.state else None
            lines.append(f"R{r.round} ({r.mode.value}): {pos} [disagreement={score}]")
    return "\n".join(lines)


def _finals_summary(finals: dict[str, AnalystPosition]) -> str:
    lines = []
    for lens, pos in finals.items():
        claims = "; ".join(f"{c.id}({c.direction.value},imp{c.importance}): {c.text} [ev: {', '.join(c.evidence_ids) or '-'}]"
                           for c in pos.claims)
        lines.append(f"{lens}: {pos.stance.value} conf={pos.confidence:.2f}. {claims}. summary: {pos.summary}")
    return "\n".join(lines)


def _consensus_ids(finals: dict[str, AnalystPosition], contested_ids: set[str]) -> list[str]:
    return [c.id for pos in finals.values() for c in pos.claims
            if c.id not in contested_ids and c.importance >= settings.consensus_importance_min]


async def synthesize(
    *, thesis: Thesis, rounds: list[RoundRecord], finals: dict[str, AnalystPosition],
    resolutions: list[Resolution], provider: LLMProvider, ledger: Ledger,
    data_as_of: str | None,
) -> CommitteeMemo:
    resolved = [r for r in resolutions if r.method == ResolutionMethod.TIEBREAKER]
    unresolved = [r for r in resolutions if r.method == ResolutionMethod.FLAG_UNRESOLVED]
    contested_ids = {r.claim_id for r in resolutions}
    consensus = _consensus_ids(finals, contested_ids)

    user = (_PROMPTS / "chair_user.md").read_text().format(
        thesis=thesis.statement, rounds=_rounds_summary(rounds), finals=_finals_summary(finals),
        resolved="\n".join(f"{r.claim_id}: {r.verdict} — {r.reasoning}" for r in resolved) or "(none)",
        unresolved="\n".join(f"{r.claim_id}: {r.reasoning}" for r in unresolved) or "(none)",
        consensus=", ".join(consensus) or "(none)",
    )
    system = (_PROMPTS / "chair_system.md").read_text()
    reservation = ledger.reserve(Pool.SYNTHESIS, "chair", "synthesis", ledger.remaining(Pool.SYNTHESIS))
    out_cap = max(settings.min_output_tokens, reservation.tokens - estimate_tokens(system + user))
    try:
        memo, usage = await provider.structured(
            schema=CommitteeMemo, system=system,
            user=user, tier=Tier.PRO, max_tokens=out_cap, kind="synthesis",
        )
        ledger.commit(reservation, usage)
    except Exception:
        ledger.release(reservation)
        raise

    # deterministic fields come from state, not from the model
    memo.verdicts = [AnalystVerdict(lens=lens, stance=pos.stance, confidence=pos.confidence,
                                    summary=pos.summary or (pos.claims[0].text if pos.claims else ""))
                     for lens, pos in finals.items()]
    memo.consensus_claim_ids = consensus
    memo.resolved, memo.unresolved = resolved, unresolved
    memo.data_as_of = data_as_of
    memo.budget_summary = ledger.summary()
    known_claims = {c.id for pos in finals.values() for c in pos.claims}
    memo.reasoning_trace = [t for t in memo.reasoning_trace if t.claim_id is None or t.claim_id in known_claims]
    return memo
