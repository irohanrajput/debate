from __future__ import annotations

import json
from typing import Any, Callable

import numpy as np
import pandas as pd

from committee.config import settings
from committee.evidence.market import MarketSnapshot
from committee.evidence.store import EvidenceStore
from committee.models import Evidence

ToolFn = Callable[..., list[Evidence]]

TOOL_DESCRIPTIONS: dict[str, str] = {
    "search_corpus": "Semantic search over the ingested text corpus (filings, facts, past decisions, docs). args: query (str), entity (str, optional), min_reliability (float 0-1, optional)",
    "company_snapshot": "Fundamentals for a ticker: growth, margins, valuation multiples, balance sheet. args: ticker (str)",
    "price_stats": "Price statistics from full daily history: trend vs moving averages, volatility, drawdown, momentum, relative strength. args: ticker (str)",
    "peer_compare": "Compare valuation/growth across tickers. args: ticker (str), peers (list of str)",
    "macro_context": "Market regime: index returns, rates, volatility. args: none",
}


def _fmt(value: Any, pct: bool = False) -> str:
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return "n/a"
    return f"{value * 100:.1f}%" if pct else (f"{value:,.2f}" if isinstance(value, float) else str(value))


def _returns(close: pd.Series, days: int) -> float | None:
    if len(close) <= days:
        return None
    return float(close.iloc[-1] / close.iloc[-1 - days] - 1)


class ToolBox:
    """The shared tool set. Agents choose which tools to call and with what
    args; they never define tools. Results are Evidence, cached per run."""

    def __init__(self, snapshot: MarketSnapshot, corpus: Any, store: EvidenceStore) -> None:
        self._snap = snapshot
        self._corpus = corpus
        self._store = store
        self._tools: dict[str, ToolFn] = {
            "search_corpus": self.search_corpus,
            "company_snapshot": self.company_snapshot,
            "price_stats": self.price_stats,
            "peer_compare": self.peer_compare,
            "macro_context": self.macro_context,
        }

    def names(self) -> list[str]:
        return list(self._tools)

    def call(self, name: str, fetched_by: str, **args: Any) -> list[Evidence]:
        if name not in self._tools:
            return [self._error(f"unknown tool {name}", fetched_by)]
        try:
            evidence = self._tools[name](**args)
        except TypeError as exc:
            return [self._error(f"bad args for {name}: {exc}", fetched_by)]
        except Exception as exc:
            return [self._error(f"{name} failed: {exc}", fetched_by)]
        for ev in evidence:
            ev.fetched_by = ev.fetched_by or fetched_by
        return evidence

    def _error(self, message: str, fetched_by: str) -> Evidence:
        return self._store.register(source="tool_error", ref=message[:80], snippet=message, fetched_by=fetched_by)

    def search_corpus(self, query: str, entity: str | None = None,
                      min_reliability: float | None = None) -> list[Evidence]:
        docs = self._corpus.search(query, entity=entity, min_reliability=min_reliability)
        return [self._store.register(
            source="corpus", ref=str(d.metadata.get("record_id", "")), snippet=d.page_content,
            as_of=str(d.metadata.get("timestamp") or "") or None,
            reliability=d.metadata.get("reliability") or None,
        ) for d in docs]

    def company_snapshot(self, ticker: str) -> list[Evidence]:
        ticker = ticker.upper()
        if not self._snap.ensure(ticker):
            return [self._error(f"no market data for {ticker}", "")]
        info = self._snap.info.get(ticker, {})
        income = self._snap.statements.get(ticker, {}).get("income")
        growth = None
        if income is not None and "Total Revenue" in income.index and income.shape[1] >= 2:
            rev = income.loc["Total Revenue"].dropna()
            if len(rev) >= 2 and rev.iloc[1]:
                growth = float(rev.iloc[0] / rev.iloc[1] - 1)
        fields = {
            "market_cap": _fmt(info.get("marketCap")),
            "revenue_growth_yoy": _fmt(growth if growth is not None else info.get("revenueGrowth"), pct=True),
            "gross_margin": _fmt(info.get("grossMargins"), pct=True),
            "operating_margin": _fmt(info.get("operatingMargins"), pct=True),
            "net_margin": _fmt(info.get("profitMargins"), pct=True),
            "trailing_pe": _fmt(info.get("trailingPE")),
            "forward_pe": _fmt(info.get("forwardPE")),
            "ev_to_ebitda": _fmt(info.get("enterpriseToEbitda")),
            "roe": _fmt(info.get("returnOnEquity"), pct=True),
            "debt_to_equity": _fmt(info.get("debtToEquity")),
            "free_cash_flow": _fmt(info.get("freeCashflow")),
        }
        snippet = f"{ticker} fundamentals: " + ", ".join(f"{k}={v}" for k, v in fields.items())
        return [self._store.register(source="yfinance", ref=f"snapshot:{ticker}",
                                     snippet=snippet[: settings.tool_result_char_cap], as_of=self._snap.as_of)]

    def price_stats(self, ticker: str) -> list[Evidence]:
        ticker = ticker.upper()
        if not self._snap.ensure(ticker):
            return [self._error(f"no market data for {ticker}", "")]
        close = self._snap.history[ticker]["Close"].dropna()
        window = close.tail(settings.drawdown_window_days)
        drawdown = float((window / window.cummax() - 1).min())
        daily = close.pct_change().dropna().tail(settings.vol_window_days)
        vol = float(daily.std() * np.sqrt(settings.trading_days_per_year))
        ma_s = float(close.tail(settings.ma_short).mean())
        ma_l = float(close.tail(settings.ma_long).mean())
        ma_l_std = float(close.tail(settings.ma_long).std())
        z = (float(close.iloc[-1]) - ma_l) / ma_l_std if ma_l_std else None
        rs = None
        bench = self._snap.history.get(settings.macro_tickers[0])
        if bench is not None:
            r_t = _returns(close, settings.relative_strength_days)
            r_b = _returns(bench["Close"].dropna(), settings.relative_strength_days)
            if r_t is not None and r_b is not None:
                rs = r_t - r_b
        fields = {
            "last_close": _fmt(float(close.iloc[-1])),
            "return_6m": _fmt(_returns(close, settings.relative_strength_days), pct=True),
            "return_1y": _fmt(_returns(close, settings.trading_days_per_year), pct=True),
            "vol_30d_annualized": _fmt(vol, pct=True),
            "max_drawdown_1y": _fmt(drawdown, pct=True),
            f"vs_{settings.ma_short}dma": _fmt(float(close.iloc[-1]) / ma_s - 1, pct=True),
            f"vs_{settings.ma_long}dma": _fmt(float(close.iloc[-1]) / ma_l - 1, pct=True),
            f"zscore_vs_{settings.ma_long}dma": _fmt(z),
            "relative_strength_6m_vs_index": _fmt(rs, pct=True),
        }
        snippet = f"{ticker} price stats: " + ", ".join(f"{k}={v}" for k, v in fields.items())
        return [self._store.register(source="yfinance", ref=f"price_stats:{ticker}",
                                     snippet=snippet[: settings.tool_result_char_cap], as_of=self._snap.as_of)]

    def peer_compare(self, ticker: str, peers: list[str]) -> list[Evidence]:
        rows = []
        for t in [ticker, *peers][:6]:
            t = t.upper()
            if not self._snap.ensure(t):
                continue
            info = self._snap.info.get(t, {})
            rows.append(f"{t}: fPE={_fmt(info.get('forwardPE'))}, ev/ebitda={_fmt(info.get('enterpriseToEbitda'))}, "
                        f"rev_growth={_fmt(info.get('revenueGrowth'), pct=True)}, gm={_fmt(info.get('grossMargins'), pct=True)}")
        if not rows:
            return [self._error("no peer data", "")]
        snippet = "peer comparison: " + " | ".join(rows)
        return [self._store.register(source="yfinance", ref=f"peers:{ticker}:{','.join(peers)}",
                                     snippet=snippet[: settings.tool_result_char_cap], as_of=self._snap.as_of)]

    def macro_context(self) -> list[Evidence]:
        parts = []
        vix_level = None
        spx_3m = None
        for t in settings.macro_tickers:
            hist = self._snap.history.get(t)
            if hist is None:
                continue
            close = hist["Close"].dropna()
            r3m = _returns(close, settings.trading_days_per_year // 4)
            level = float(close.iloc[-1])
            parts.append(f"{t}: level={_fmt(level)}, 3m={_fmt(r3m, pct=True)}")
            if t == "^VIX":
                vix_level = level
            if t == settings.macro_tickers[0]:
                spx_3m = r3m
        regime = "unknown"
        if vix_level is not None and spx_3m is not None:
            regime = "risk_on" if vix_level < settings.risk_on_vix_max and spx_3m > 0 else "risk_off_or_mixed"
        snippet = f"macro context (regime={regime}): " + " | ".join(parts)
        return [self._store.register(source="yfinance", ref="macro", snippet=snippet[: settings.tool_result_char_cap],
                                     as_of=self._snap.as_of)]


def tool_catalog() -> str:
    return json.dumps(TOOL_DESCRIPTIONS, indent=2)
