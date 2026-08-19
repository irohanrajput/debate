from committee.debate.budget import Ledger, Pool
from committee.debate.policies import get_policy
from committee.models import ContestedClaim, DebateState, Mode, Tier

LENSES = ["fundamentalist", "momentum", "quality", "risk"]


def _state(contested: list[ContestedClaim], score: float = 0.5) -> DebateState:
    return DebateState(round=2, mode=Mode.EXPLORE, disagreement_score=score, contested=contested)


def test_round_one_is_discover():
    decision = get_policy("explore_exploit").allocate(None, Ledger(total=40000), 1, LENSES)
    assert decision.mode == Mode.DISCOVER
    assert decision.selected_lenses == LENSES
    assert all(t == Tier.FLASH for t in decision.tier_by_lens.values())


def test_contested_claims_trigger_exploit_with_pro_tier():
    contested = [ContestedClaim(claim_id="FUND-R2-1", owner="fundamentalist",
                                against_lenses=["momentum"], score=0.9)]
    ledger = Ledger(total=40000)
    decision = get_policy("explore_exploit").allocate(_state(contested), ledger, 3, LENSES)
    assert decision.mode == Mode.EXPLOIT
    assert set(decision.selected_lenses) == {"fundamentalist", "momentum"}
    assert all(t == Tier.PRO for t in decision.tier_by_lens.values())
    assert ledger.remaining(Pool.RESEARCH) == 0  # folded into debate pool
    assert decision.transfers


def test_no_contested_means_explore():
    decision = get_policy("explore_exploit").allocate(_state([]), Ledger(total=40000), 3, LENSES)
    assert decision.mode == Mode.EXPLORE
    assert decision.selected_lenses == LENSES


def test_uniform_baseline_never_exploits():
    contested = [ContestedClaim(claim_id="X", owner="quality", against_lenses=["risk"], score=0.9)]
    decision = get_policy("uniform").allocate(_state(contested), Ledger(total=40000), 3, LENSES)
    assert decision.mode == Mode.EXPLORE
    assert all(t == Tier.FLASH for t in decision.tier_by_lens.values())
