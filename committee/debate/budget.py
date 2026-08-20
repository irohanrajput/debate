from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable

from committee.config import settings
from committee.models import Usage


class Pool(StrEnum):
    RESEARCH = "research"
    DEBATE = "debate"
    SYNTHESIS = "synthesis"


class BudgetError(Exception):
    pass


class InsufficientBudget(Exception):
    """Raised when an allocation cannot cover prompt input plus a minimal output."""


# conservative chars-to-tokens estimate used before reserving
def estimate_tokens(text: str) -> int:
    return len(text) // settings.chars_per_token


@dataclass
class Reservation:
    pool: Pool
    lens: str
    kind: str
    tokens: int
    committed: bool = False


@dataclass
class Ledger:
    """Hard token ledger. Reserve before every call; commit actual usage after.

    Synthesis pool is writable only via kind='synthesis'/'tiebreak' calls made
    by the orchestrator; analyst kinds cannot reserve from it.
    """

    total: int
    on_event: Callable[[str, dict[str, Any]], None] | None = None
    pools: dict[Pool, int] = field(default_factory=dict)
    spent: dict[Pool, int] = field(default_factory=lambda: {p: 0 for p in Pool})
    reserved: dict[Pool, int] = field(default_factory=lambda: {p: 0 for p in Pool})
    history: list[dict[str, Any]] = field(default_factory=list)

    ANALYST_POOLS = (Pool.RESEARCH, Pool.DEBATE)

    def __post_init__(self) -> None:
        self.pools = {
            Pool.RESEARCH: int(self.total * settings.research_pool_frac),
            Pool.DEBATE: int(self.total * settings.debate_pool_frac),
            Pool.SYNTHESIS: int(self.total * settings.synthesis_pool_frac),
        }
        min_viable = settings.argue_floor_tokens + settings.plan_tokens
        if self.pools[Pool.DEBATE] < min_viable:
            raise BudgetError(f"budget {self.total} too small: debate pool {self.pools[Pool.DEBATE]} < {min_viable}")

    # what's left in a wallet after spends and outstanding reservations
    def remaining(self, pool: Pool) -> int:
        return self.pools[pool] - self.spent[pool] - self.reserved[pool]

    # claim tokens before a call; truncated to what's left, never over
    def reserve(self, pool: Pool, lens: str, kind: str, tokens: int) -> Reservation:
        if pool == Pool.SYNTHESIS and kind != "synthesis":
            raise BudgetError(f"kind={kind} may not draw from synthesis pool")
        granted = max(0, min(tokens, self.remaining(pool)))
        self.reserved[pool] += granted
        res = Reservation(pool=pool, lens=lens, kind=kind, tokens=granted)
        self._log("reserve", pool=pool, lens=lens, kind=kind, requested=tokens, granted=granted)
        return res

    # record the provider's real token count and free the reservation
    def commit(self, res: Reservation, usage: Usage) -> None:
        if res.committed:
            raise BudgetError("reservation already committed")
        res.committed = True
        self.reserved[res.pool] -= res.tokens
        self.spent[res.pool] += usage.total
        self._log("commit", pool=res.pool, lens=res.lens, kind=res.kind,
                  reserved=res.tokens, actual=usage.total, model=usage.model)

    # give back an unused reservation (call skipped or failed)
    def release(self, res: Reservation) -> None:
        if not res.committed:
            res.committed = True
            self.reserved[res.pool] -= res.tokens
            self._log("release", pool=res.pool, lens=res.lens, kind=res.kind, tokens=res.tokens)

    # policy-driven move between wallets (e.g. research folded into debate at exploit)
    def transfer(self, src: Pool, dst: Pool, tokens: int) -> int:
        if Pool.SYNTHESIS in (src, dst):
            raise BudgetError("synthesis pool cannot participate in transfers")
        moved = max(0, min(tokens, self.remaining(src)))
        self.pools[src] -= moved
        self.pools[dst] += moved
        self._log("transfer", src=src, dst=dst, tokens=moved)
        return moved

    # actual tokens consumed across all pools
    def total_spent(self) -> int:
        return sum(self.spent.values())

    # per-pool spend snapshot for events and the memo
    def summary(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "spent": self.total_spent(),
            "pools": {p.value: {"size": self.pools[p], "spent": self.spent[p], "remaining": self.remaining(p)} for p in Pool},
        }

    # every ledger movement is recorded for the trace
    def _log(self, op: str, **kw: Any) -> None:
        entry = {"op": op, **{k: (v.value if isinstance(v, Pool) else v) for k, v in kw.items()}}
        self.history.append(entry)
        if self.on_event:
            self.on_event(op, entry)
