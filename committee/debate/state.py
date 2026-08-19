from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

from committee.agents.analyst import Analyst
from committee.debate.budget import Ledger
from committee.debate.disagreement import Embedder
from committee.debate.policies.base import BudgetPolicy
from committee.evidence.store import EvidenceStore
from committee.evidence.tools import ToolBox
from committee.llm.client import LLMProvider
from committee.models import (
    AnalystMemory,
    BudgetDecision,
    CommitteeMemo,
    ContradictionVerdict,
    DebateState,
    Resolution,
    RoundRecord,
    Thesis,
)
from committee.trace.events import EventBus


@dataclass
class DebateRuntime:
    run_id: str
    thesis: Thesis
    lenses: list[str]
    policy: BudgetPolicy
    provider: LLMProvider
    embedder: Embedder
    toolbox: ToolBox
    store: EvidenceStore
    ledger: Ledger
    bus: EventBus
    data_as_of: str | None = None
    round: int = 0
    memories: dict[str, AnalystMemory] = field(default_factory=dict)
    analysts: dict[str, Analyst] = field(default_factory=dict)
    rounds: list[RoundRecord] = field(default_factory=list)
    decision: BudgetDecision | None = None
    state: DebateState | None = None
    judge_cache: dict[tuple[str, str], ContradictionVerdict] = field(default_factory=dict)
    resolutions: list[Resolution] = field(default_factory=list)
    memo: CommitteeMemo | None = None


class GraphState(TypedDict):
    rt: DebateRuntime
