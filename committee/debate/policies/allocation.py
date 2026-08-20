from __future__ import annotations

from committee.config import settings
from committee.debate.budget import Ledger, Pool
from committee.models import Tier


# equal share per lens with an optional minimum
def split_evenly(amount: int, lenses: list[str], floor: int = 0) -> dict[str, int]:
    per = max(floor, int(amount / max(1, len(lenses))))
    return {lens: per for lens in lenses}


# share proportional to contested involvement
def split_weighted(amount: int, weights: dict[str, int], floor: int | None = None) -> dict[str, int]:
    total_w = sum(weights.values()) or 1
    floor = floor if floor is not None else settings.argue_floor_tokens
    return {lens: max(floor, int(amount * w / total_w)) for lens, w in weights.items()}


# same model tier for everyone
def uniform_tiers(lenses: list[str], tier: Tier) -> dict[str, Tier]:
    return {lens: tier for lens in lenses}


# spend a fraction of whatever is left in a pool
def share_of_remaining(ledger: Ledger, pool: Pool, frac: float) -> int:
    return int(ledger.remaining(pool) * frac)
