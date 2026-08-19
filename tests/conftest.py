from committee.models import (
    AnalystPosition,
    Claim,
    CommitteeMemo,
    ContradictionVerdict,
    Direction,
    Findings,
    Observation,
    Recommendation,
    ResearchPlan,
    Response,
    ResponseAction,
    Stance,
)

CONTESTED_TEXT = "the current valuation is stretched relative to history"
ALL_R2_IDS = ["FUND-R2-1", "MOME-R2-1", "QUAL-R2-1", "RISK-R2-1"]


def plan() -> ResearchPlan:
    return ResearchPlan(queries=[], rationale="no research needed in test")


def findings() -> Findings:
    return Findings(observations=[Observation(text="observed something")],
                    open_questions=["what is fair value?"], summary="initial look")


def position(stance: Stance, text: str, direction: Direction, responses: list[Response] = ()) -> AnalystPosition:
    return AnalystPosition(stance=stance, confidence=0.7,
                           claims=[Claim(text=text, direction=direction, importance=4)],
                           responses=list(responses), summary=f"{stance.value} case")


def responses_for_all() -> list[Response]:
    return [Response(claim_id=cid, action=ResponseAction.PARTIAL, text="addressed") for cid in ALL_R2_IDS]


def verdict_contradiction() -> ContradictionVerdict:
    return ContradictionVerdict(contradicts=True, score=0.9, why="same subject, opposite direction")


def memo() -> CommitteeMemo:
    return CommitteeMemo(recommendation=Recommendation.BUY, headline="test memo", confidence=0.6, verdicts=[])


def full_scripts() -> dict:
    return {
        "plan": [plan() for _ in range(8)],
        "findings": [findings() for _ in range(4)],
        "argue": [
            position(Stance.SELL, CONTESTED_TEXT, Direction.BEAR),
            position(Stance.BUY, CONTESTED_TEXT, Direction.BULL),
            position(Stance.BUY, "market leadership is durable", Direction.BULL),
            position(Stance.PASS, "macro regime is hostile", Direction.BEAR),
            position(Stance.BUY, "converged view a", Direction.BULL, responses_for_all()),
            position(Stance.BUY, "converged view b", Direction.BULL, responses_for_all()),
        ],
        "judge": [verdict_contradiction() for _ in range(4)],
        "synthesis": [memo()],
    }
