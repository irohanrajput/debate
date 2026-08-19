import pytest
from pydantic import ValidationError

from committee.models import (
    AnalystPosition,
    Claim,
    CommitteeMemo,
    Direction,
    Recommendation,
    Response,
    ResponseAction,
    Stance,
)


def test_claim_importance_bounds():
    with pytest.raises(ValidationError):
        Claim(text="x", direction=Direction.BULL, importance=6)


def test_confidence_bounds():
    with pytest.raises(ValidationError):
        AnalystPosition(stance=Stance.BUY, confidence=1.5, claims=[])


def test_buy_limited_requires_guidance():
    with pytest.raises(ValidationError):
        CommitteeMemo(recommendation=Recommendation.BUY_LIMITED, headline="h", confidence=0.5, verdicts=[])
    memo = CommitteeMemo(recommendation=Recommendation.BUY_LIMITED, headline="h", confidence=0.5,
                         verdicts=[], position_guidance="1/3 size, add on pullback")
    assert memo.position_guidance


def test_missing_responses_detection():
    pos = AnalystPosition(stance=Stance.HOLD, confidence=0.5, claims=[],
                          responses=[Response(claim_id="A-1", action=ResponseAction.REBUT, text="no")])
    assert pos.missing_responses(["A-1", "B-2"]) == ["B-2"]
