import pytest

from committee.config import settings
from committee.debate.budget import BudgetError, Ledger, Pool
from committee.models import Usage


def test_pools_sum_to_total_fractions():
    ledger = Ledger(total=40000)
    assert ledger.pools[Pool.RESEARCH] == int(40000 * settings.research_pool_frac)
    assert ledger.pools[Pool.SYNTHESIS] == int(40000 * settings.synthesis_pool_frac)


def test_reserve_truncates_to_remaining():
    ledger = Ledger(total=40000)
    res = ledger.reserve(Pool.RESEARCH, "a", "plan", 10**9)
    assert res.tokens == ledger.pools[Pool.RESEARCH]
    assert ledger.remaining(Pool.RESEARCH) == 0


def test_commit_and_release_restore_reservation():
    ledger = Ledger(total=40000)
    res = ledger.reserve(Pool.DEBATE, "a", "argue", 1000)
    ledger.commit(res, Usage(input_tokens=300, output_tokens=200))
    assert ledger.spent[Pool.DEBATE] == 500
    res2 = ledger.reserve(Pool.DEBATE, "b", "argue", 1000)
    ledger.release(res2)
    assert ledger.reserved[Pool.DEBATE] == 0


def test_double_commit_raises():
    ledger = Ledger(total=40000)
    res = ledger.reserve(Pool.DEBATE, "a", "argue", 100)
    ledger.commit(res, Usage(output_tokens=10))
    with pytest.raises(BudgetError):
        ledger.commit(res, Usage(output_tokens=10))


def test_synthesis_pool_locked_to_synthesis_kind():
    ledger = Ledger(total=40000)
    for kind in ("argue", "plan", "judge", "tiebreak"):
        with pytest.raises(BudgetError):
            ledger.reserve(Pool.SYNTHESIS, "a", kind, 100)
    assert ledger.reserve(Pool.SYNTHESIS, "chair", "synthesis", 100).tokens == 100


def test_synthesis_pool_excluded_from_transfers():
    ledger = Ledger(total=40000)
    with pytest.raises(BudgetError):
        ledger.transfer(Pool.SYNTHESIS, Pool.DEBATE, 100)


def test_too_small_budget_rejected():
    with pytest.raises(BudgetError):
        Ledger(total=100)
