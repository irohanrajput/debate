from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from committee.config import settings
from committee.debate.budget import Ledger, Pool, estimate_tokens
from committee.llm.client import LLMProvider
from committee.models import (
    AnalystPosition,
    ContestedClaim,
    Resolution,
    ResolutionMethod,
    Tier,
)

_PROMPTS = Path(__file__).parent.parent / "agents" / "prompts"


class TieBreak(BaseModel):
    verdict: str
    reasoning: str


def _claim_context(claim_id: str, positions: dict[str, AnalystPosition]) -> tuple[str, str]:
    for pos in positions.values():
        for c in pos.claims:
            if c.id == claim_id:
                return c.text, ", ".join(c.evidence_ids) or "none cited"
    return "(claim text unavailable)", "none"


def _objections(claim_id: str, against: list[str], positions: dict[str, AnalystPosition]) -> str:
    lines = []
    for lens in against:
        pos = positions.get(lens)
        if not pos:
            continue
        for r in pos.responses:
            if r.claim_id == claim_id:
                lines.append(f"{lens} [{r.action.value}]: {r.text}")
        for c in pos.claims:
            lines.append(f"{lens} claim {c.id}: {c.text}")
    return "\n".join(lines) or "(no explicit objection text)"


async def resolve_conflicts(
    *, contested: list[ContestedClaim], positions: dict[str, AnalystPosition],
    provider: LLMProvider, ledger: Ledger,
) -> list[Resolution]:
    """Tie-break each surviving contested claim if budget allows; else flag.
    Never averages."""
    resolutions: list[Resolution] = []
    template = (_PROMPTS / "tiebreak_user.md").read_text()
    system = "You are a neutral senior investment adjudicator."
    ranked = sorted(contested, key=lambda c: -c.score)
    for i, claim in enumerate(ranked):
        if i >= settings.max_tiebreaks:
            resolutions.append(Resolution(
                claim_id=claim.claim_id, method=ResolutionMethod.FLAG_UNRESOLVED,
                reasoning=f"contested by {claim.against_lenses} vs {claim.owner}; beyond tiebreak cap ({settings.max_tiebreaks}), flagged for human review",
            ))
            continue
        text, evidence = _claim_context(claim.claim_id, positions)
        user = template.format(owner=claim.owner, claim_text=text, owner_evidence=evidence,
                               against=", ".join(claim.against_lenses),
                               objections=_objections(claim.claim_id, claim.against_lenses, positions))
        need = estimate_tokens(system + user) + settings.schema_overhead_tokens + settings.tiebreak_output_tokens
        res = ledger.reserve(Pool.DEBATE, "tiebreaker", "tiebreak", need)
        if res.tokens < need:
            ledger.release(res)
            resolutions.append(Resolution(
                claim_id=claim.claim_id, method=ResolutionMethod.FLAG_UNRESOLVED,
                reasoning=f"contested by {claim.against_lenses} vs {claim.owner}; no budget left to adjudicate",
            ))
            continue
        try:
            verdict, usage = await provider.structured(
                schema=TieBreak, system=system,
                user=user, tier=Tier.PRO, max_tokens=settings.tiebreak_output_tokens, kind="tiebreak",
            )
            ledger.commit(res, usage)
            resolutions.append(Resolution(claim_id=claim.claim_id, method=ResolutionMethod.TIEBREAKER,
                                          verdict=verdict.verdict, reasoning=verdict.reasoning, usage=usage))
        except Exception as exc:
            ledger.release(res)
            resolutions.append(Resolution(claim_id=claim.claim_id, method=ResolutionMethod.FLAG_UNRESOLVED,
                                          reasoning=f"tie-break failed ({exc}); flagged"))
    return resolutions
