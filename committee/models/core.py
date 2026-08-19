from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class Stance(StrEnum):
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    PASS = "PASS"


class Direction(StrEnum):
    BULL = "BULL"
    BEAR = "BEAR"
    NEUTRAL = "NEUTRAL"


class Mode(StrEnum):
    DISCOVER = "DISCOVER"
    EXPLORE = "EXPLORE"
    EXPLOIT = "EXPLOIT"


class Tier(StrEnum):
    FLASH = "flash"
    PRO = "pro"


class ResponseAction(StrEnum):
    CONCEDE = "CONCEDE"
    PARTIAL = "PARTIAL"
    REBUT = "REBUT"
    INCORPORATE = "INCORPORATE"


class ResolutionMethod(StrEnum):
    TIEBREAKER = "TIEBREAKER"
    FLAG_UNRESOLVED = "FLAG_UNRESOLVED"


class Recommendation(StrEnum):
    BUY = "BUY"
    BUY_LIMITED = "BUY_LIMITED"
    WAIT = "WAIT"
    DO_NOT_BUY = "DO_NOT_BUY"


class Thesis(BaseModel):
    statement: str
    ticker: str | None = None
    entity: str | None = None
    horizon: str | None = None
    constraints: list[str] = Field(default_factory=list)


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    kind: str = ""

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


class Evidence(BaseModel):
    id: str
    source: str
    ref: str
    snippet: str
    as_of: str | None = None
    reliability: float | None = None
    fetched_by: str | None = None


class ToolRequest(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class ResearchPlan(BaseModel):
    queries: list[ToolRequest]
    rationale: str


class Observation(BaseModel):
    text: str
    evidence_ids: list[str] = Field(default_factory=list)


class Findings(BaseModel):
    lens: str = ""
    round: int = 0
    observations: list[Observation]
    open_questions: list[str] = Field(default_factory=list)
    what_would_change_my_view: list[str] = Field(default_factory=list)
    provisional_lean: Direction | None = None
    summary: str = ""
    usage: Usage | None = None


class Claim(BaseModel):
    id: str = ""
    text: str
    direction: Direction
    importance: int = Field(ge=1, le=5)
    evidence_ids: list[str] = Field(default_factory=list)


class Response(BaseModel):
    claim_id: str
    action: ResponseAction
    text: str
    evidence_ids: list[str] = Field(default_factory=list)


class AnalystPosition(BaseModel):
    lens: str = ""
    round: int = 0
    stance: Stance
    confidence: float = Field(ge=0.0, le=1.0)
    claims: list[Claim]
    responses: list[Response] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    changed_from_prior: bool = False
    change_reason: str = ""
    summary: str = ""
    usage: Usage | None = None

    def missing_responses(self, must_address: list[str]) -> list[str]:
        answered = {r.claim_id for r in self.responses}
        return [cid for cid in must_address if cid not in answered]


class ContestedClaim(BaseModel):
    claim_id: str
    owner: str
    for_lenses: list[str] = Field(default_factory=list)
    against_lenses: list[str] = Field(default_factory=list)
    score: float = 0.0
    rounds_contested: int = 1


class ContradictionVerdict(BaseModel):
    contradicts: bool
    score: float = Field(ge=0.0, le=1.0)
    why: str


class DebateState(BaseModel):
    round: int
    mode: Mode
    stance_counts: dict[str, int] = Field(default_factory=dict)
    disagreement_score: float | None = None
    convergence_delta: float | None = None
    contested: list[ContestedClaim] = Field(default_factory=list)
    converged: bool = False
    coverage_notes: list[str] = Field(default_factory=list)


class BudgetDecision(BaseModel):
    round: int
    mode: Mode
    selected_lenses: list[str]
    tier_by_lens: dict[str, Tier] = Field(default_factory=dict)
    research_by_lens: dict[str, int] = Field(default_factory=dict)
    argue_by_lens: dict[str, int] = Field(default_factory=dict)
    transfers: list[dict[str, Any]] = Field(default_factory=list)
    rationale: str = ""
    state_snapshot: DebateState | None = None


class Resolution(BaseModel):
    claim_id: str
    method: ResolutionMethod
    verdict: str = ""
    reasoning: str
    usage: Usage | None = None


class AnalystVerdict(BaseModel):
    lens: str
    stance: Stance
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str


class TraceRef(BaseModel):
    round: int
    lens: str
    claim_id: str | None = None
    note: str


class CommitteeMemo(BaseModel):
    recommendation: Recommendation
    headline: str
    confidence: float = Field(ge=0.0, le=1.0)
    position_guidance: str = ""
    verdicts: list[AnalystVerdict]
    consensus_claim_ids: list[str] = Field(default_factory=list)
    resolved: list[Resolution] = Field(default_factory=list)
    unresolved: list[Resolution] = Field(default_factory=list)
    reasoning_trace: list[TraceRef] = Field(default_factory=list)
    budget_summary: dict[str, Any] = Field(default_factory=dict)
    data_as_of: str | None = None

    @model_validator(mode="after")
    def _limited_needs_guidance(self) -> "CommitteeMemo":
        if self.recommendation == Recommendation.BUY_LIMITED and not self.position_guidance:
            raise ValueError("BUY_LIMITED requires position_guidance")
        return self


class AnalystMemory(BaseModel):
    lens: str
    findings: Findings | None = None
    positions: list[AnalystPosition] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    pending_must_address: list[str] = Field(default_factory=list)

    @property
    def latest_position(self) -> AnalystPosition | None:
        return self.positions[-1] if self.positions else None


class TraceEvent(BaseModel):
    ts: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    run_id: str
    type: str
    round: int | None = None
    lens: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class RoundRecord(BaseModel):
    round: int
    mode: Mode
    budget_decision: BudgetDecision
    findings: dict[str, Findings] = Field(default_factory=dict)
    positions: dict[str, AnalystPosition] = Field(default_factory=dict)
    state: DebateState | None = None


class DebateTrace(BaseModel):
    run_id: str
    thesis: Thesis
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    rounds: list[RoundRecord] = Field(default_factory=list)
    resolutions: list[Resolution] = Field(default_factory=list)
    memo: CommitteeMemo | None = None
    totals: dict[str, Any] = Field(default_factory=dict)
