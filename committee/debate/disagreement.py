from __future__ import annotations

import math
from typing import Protocol

import numpy as np

from committee.config import settings
from committee.debate.budget import Ledger, Pool, estimate_tokens
from committee.llm.client import LLMProvider
from committee.models import (
    AnalystPosition,
    Claim,
    ContestedClaim,
    ContradictionVerdict,
    DebateState,
    Direction,
    Mode,
    ResponseAction,
    Tier,
)

_JUDGE_SYSTEM = "You judge whether two investment claims genuinely contradict each other (not merely differ in topic or emphasis)."
_OPPOSED = {Direction.BULL: Direction.BEAR, Direction.BEAR: Direction.BULL}


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


def _stance_entropy(positions: dict[str, AnalystPosition]) -> float:
    counts: dict[str, int] = {}
    for pos in positions.values():
        counts[pos.stance.value] = counts.get(pos.stance.value, 0) + 1
    n = sum(counts.values())
    if n <= 1:
        return 0.0
    entropy = -sum((c / n) * math.log(c / n) for c in counts.values())
    return entropy / math.log(min(n, 4))


def _cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb)) or 1.0
    return float(va @ vb / denom)


def _candidate_pairs(positions: dict[str, AnalystPosition], embedder: Embedder) -> list[tuple[str, Claim, str, Claim]]:
    owned = [(lens, c) for lens, pos in positions.items() for c in pos.claims]
    if len(owned) < 2:
        return []
    vectors = embedder.embed_documents([c.text for _, c in owned])
    pairs = []
    for i in range(len(owned)):
        for j in range(i + 1, len(owned)):
            (lens_a, a), (lens_b, b) = owned[i], owned[j]
            if lens_a == lens_b or _OPPOSED.get(a.direction) != b.direction:
                continue
            if _cosine(vectors[i], vectors[j]) >= settings.similarity_candidate_threshold:
                pairs.append((lens_a, a, lens_b, b))
    return pairs


def _rebuttal_pairs(positions: dict[str, AnalystPosition]) -> list[tuple[str, str, str]]:
    """(rebutting_lens, target_claim_id, rebutting_claim_or_'') from explicit REBUT responses."""
    return [(lens, r.claim_id, "") for lens, pos in positions.items()
            for r in pos.responses if r.action == ResponseAction.REBUT]


async def assess(
    *, round: int, positions: dict[str, AnalystPosition], prior: DebateState | None,
    embedder: Embedder, provider: LLMProvider, ledger: Ledger,
    judge_cache: dict[tuple[str, str], ContradictionVerdict],
) -> DebateState:
    entropy = _stance_entropy(positions)
    claim_owner = {c.id: lens for lens, pos in positions.items() for c in pos.claims}
    contested: dict[str, ContestedClaim] = {}

    for lens_a, a, lens_b, b in _candidate_pairs(positions, embedder):
        key = tuple(sorted((a.id, b.id)))
        verdict = judge_cache.get(key)
        if verdict is None:
            judge_user = f"Claim A ({lens_a}): {a.text}\nClaim B ({lens_b}): {b.text}\nDo they contradict?"
            need = estimate_tokens(_JUDGE_SYSTEM + judge_user) + settings.schema_overhead_tokens + settings.judge_output_tokens
            res = ledger.reserve(Pool.DEBATE, "judge", "judge", need)
            if res.tokens < need:
                ledger.release(res)
                continue
            try:
                verdict, usage = await provider.structured(
                    schema=ContradictionVerdict, system=_JUDGE_SYSTEM,
                    user=judge_user,
                    tier=Tier.FLASH, max_tokens=settings.judge_output_tokens, kind="judge",
                )
                ledger.commit(res, usage)
            except Exception:
                ledger.release(res)
                continue
            judge_cache[key] = verdict
        if verdict.contradicts and verdict.score >= settings.contradiction_threshold:
            for cid, owner, against in ((a.id, lens_a, lens_b), (b.id, lens_b, lens_a)):
                entry = contested.setdefault(cid, ContestedClaim(claim_id=cid, owner=owner, score=0.0))
                entry.score = max(entry.score, verdict.score)
                if against not in entry.against_lenses:
                    entry.against_lenses.append(against)

    for lens, target_id, _ in _rebuttal_pairs(positions):
        owner = claim_owner.get(target_id)
        if owner and owner != lens:
            entry = contested.setdefault(target_id, ContestedClaim(claim_id=target_id, owner=owner,
                                                                   score=settings.contradiction_threshold))
            if lens not in entry.against_lenses:
                entry.against_lenses.append(lens)

    if prior:
        prior_ids = {c.claim_id for c in prior.contested}
        for entry in contested.values():
            if entry.claim_id in prior_ids:
                entry.rounds_contested += 1

    total_importance = sum(c.importance for pos in positions.values() for c in pos.claims) or 1
    contested_importance = sum(
        next((c.importance for pos in positions.values() for c in pos.claims if c.id == cid), 0)
        for cid in contested
    )
    score = settings.stance_weight * entropy + settings.contested_weight * (contested_importance / total_importance)
    delta = None if prior is None or prior.disagreement_score is None else score - prior.disagreement_score
    converged = score < settings.theta_converged and (delta is None or abs(delta) < settings.convergence_delta_cap or delta < 0)

    return DebateState(
        round=round, mode=Mode.EXPLOIT if contested else Mode.EXPLORE,
        stance_counts={s: sum(1 for p in positions.values() if p.stance.value == s)
                       for s in {p.stance.value for p in positions.values()}},
        disagreement_score=round_score(score), convergence_delta=None if delta is None else round_score(delta),
        contested=sorted(contested.values(), key=lambda c: -c.score), converged=converged,
    )


def round_score(x: float) -> float:
    return float(f"{x:.4f}")


def must_address_for(lens: str, state: DebateState) -> list[str]:
    """A lens must address contested claims it owns (defend) or disputes."""
    return [c.claim_id for c in state.contested if c.owner == lens or lens in c.against_lenses]
