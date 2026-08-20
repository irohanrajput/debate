import pandas as pd
import pytest

from committee.evidence.market import MarketSnapshot
from committee.evidence.store import EvidenceStore
from committee.evidence.tools import ToolBox
from committee.evidence.corpus import NullCorpus


def _snapshot_with(ticker: str) -> MarketSnapshot:
    snap = MarketSnapshot()
    snap.offline = True
    idx = pd.date_range("2024-01-01", periods=300, freq="B")
    close = pd.Series(range(100, 400), index=idx, dtype=float)
    snap.history[ticker] = pd.DataFrame({"Close": close, "Volume": 1_000_000.0}, index=idx)
    snap.info[ticker] = {"sharesOutstanding": 1_000_000_000, "totalRevenue": 50_000_000_000}
    cols = pd.to_datetime(["2025-12-31", "2024-12-31"])
    snap.statements[ticker] = {
        "income": pd.DataFrame([[50_000_000_000, 40_000_000_000]], index=["Total Revenue"], columns=cols),
    }
    return snap


def test_scenario_math_arithmetic():
    snap = _snapshot_with("TEST")
    box = ToolBox(snap, NullCorpus(), EvidenceStore())
    [ev] = box.scenario_math("TEST", revenue_growth=0.10, net_margin=0.20, exit_multiple=20.0)
    # 50B * 1.1 * 0.2 = 11B net income; EPS 11; price 220
    assert "implied price 220.00" in ev.snippet
    assert ev.source == "yfinance"


def test_scenario_math_missing_data_is_error_evidence():
    snap = MarketSnapshot()
    snap.offline = True
    box = ToolBox(snap, NullCorpus(), EvidenceStore())
    [ev] = box.scenario_math("NOPE", revenue_growth=0.1, net_margin=0.2, exit_multiple=20)
    assert ev.source == "tool_error"


def test_entity_timeline_empty_corpus():
    box = ToolBox(_snapshot_with("TEST"), NullCorpus(), EvidenceStore())
    [ev] = box.entity_timeline("NovaTech Inc.")
    assert ev.source == "tool_error"


def test_price_stats_returns_trend_and_risk_cards():
    box = ToolBox(_snapshot_with("TEST"), NullCorpus(), EvidenceStore())
    cards = box.price_stats("TEST")
    refs = {ev.ref for ev in cards}
    assert refs == {"price_stats:TEST:trend", "price_stats:TEST:risk"}
    assert any("position_in_52w_range" in ev.snippet for ev in cards)
