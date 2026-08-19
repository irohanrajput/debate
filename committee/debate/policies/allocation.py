from __future__ import annotations

from committee.config import settings
from committee.debate.budget import Ledger, Pool
from committee.models import Tier


def split_evenly(amount: int, lenses: list[str], floor: int = 0) -> dict[str, int]:
    per = max(floor, int(amount / max(1, len(lenses))))
    return {lens: per for lens in lenses}


def split_weighted(amount: int, weights: dict[str, int]) -> dict[str, int]:
    total_w = sum(weights.values()) or 1
    return {lens: max(settings.argue_floor_tokens, int(amount * w / total_w)) for lens, w in weights.items()}


def uniform_tiers(lenses: list[str], tier: Tier) -> dict[str, Tier]:
    return {lens: tier for lens in lenses}


def share_of_remaining(ledger: Ledger, pool: Pool, frac: float) -> int:
    return int(ledger.remaining(pool) * frac)
