from __future__ import annotations

from abc import ABC, abstractmethod

from committee.debate.budget import Ledger
from committee.models import BudgetDecision, DebateState

POLICIES: dict[str, type["BudgetPolicy"]] = {}


# decorator: adding a policy = one class + this registration
def register_policy(name: str):
    def deco(cls: type[BudgetPolicy]) -> type[BudgetPolicy]:
        cls.name = name
        POLICIES[name] = cls
        return cls
    return deco


# resolve a policy by CLI name
def get_policy(name: str) -> "BudgetPolicy":
    if name not in POLICIES:
        raise KeyError(f"unknown policy '{name}'; available: {sorted(POLICIES)}")
    return POLICIES[name]()


class BudgetPolicy(ABC):
    name: str = ""

    @abstractmethod
    def allocate(self, state: DebateState | None, ledger: Ledger, round: int, lenses: list[str]) -> BudgetDecision: ...
