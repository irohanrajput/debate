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
    "entity_timeline": "Every corpus record about one entity in chronological order; use to see sequences (claims, denials, restatements) that search fragments hide. args: entity (str)",
    "company_snapshot": "Fundamentals for a ticker: valuation multiples plus quarterly revenue trajectory, margin trend over years, FCF history, share dilution, EPS. args: ticker (str)",
    "price_stats": "Price and volume statistics from full daily history: trend vs moving averages, volatility, drawdown, 52-week range position, volume trend, correlation/beta vs the index. args: ticker (str)",
    "peer_compare": "Compare valuation/growth across tickers. args: ticker (str), peers (list of str)",
    "macro_context": "Market regime: index returns, rates, volatility. args: none",
    "scenario_math": "Deterministic what-if arithmetic: project revenue at a growth rate, apply a net margin and an exit P/E, get implied price and upside/downside vs today. args: ticker (str), revenue_growth (fraction, e.g. 0.10), net_margin (fraction), exit_multiple (float)",
}


# human-readable number formatting; hides NaN/None as n/a
def _fmt(value: Any, pct: bool = False) -> str:
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return "n/a"
    return f"{value * 100:.1f}%" if pct else (f"{value:,.2f}" if isinstance(value, float) else str(value))


# simple return over the last N trading days
def _returns(close: pd.Series, days: int) -> float | None:
    if len(close) <= days:
        return None
    return float(close.iloc[-1] / close.iloc[-1 - days] - 1)


# safe row lookup in a yfinance statement table
def _row(df: pd.DataFrame | None, name: str) -> pd.Series | None:
    if df is not None and name in df.index:
        series = df.loc[name].dropna()
        if not series.empty:
            return series
    return None


class ToolBox:
    """The shared tool set. Agents choose which tools to call and with what
    args; they never define tools. Results are Evidence, cached per run."""

    def __init__(self, snapshot: MarketSnapshot, corpus: Any, store: EvidenceStore) -> None:
        self._snap = snapshot
        self._corpus = corpus
        self._store = store
        self._tools: dict[str, ToolFn] = {
            "search_corpus": self.search_corpus,
            "entity_timeline": self.entity_timeline,
            "company_snapshot": self.company_snapshot,
            "price_stats": self.price_stats,
            "peer_compare": self.peer_compare,
            "macro_context": self.macro_context,
            "scenario_math": self.scenario_math,
        }

    def names(self) -> list[str]:
        return list(self._tools)

    # single entry point for agents: bad tool, bad args, or a crash becomes error evidence, never an exception
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

    # failures are evidence too, so the agent can reason about missing data
    def _error(self, message: str, fetched_by: str) -> Evidence:
        return self._store.register(source="tool_error", ref=message[:80], snippet=message, fetched_by=fetched_by)

    # ---------- corpus ----------

    # semantic search over the text index; each hit registered with provenance
    def search_corpus(self, query: str, entity: str | None = None,
                      min_reliability: float | None = None) -> list[Evidence]:
        docs = self._corpus.search(query, entity=entity, min_reliability=min_reliability)
        return [self._store.register(
            source="corpus", ref=str(d.metadata.get("record_id", "")), snippet=d.page_content,
            as_of=str(d.metadata.get("timestamp") or "") or None,
            reliability=d.metadata.get("reliability") or None,
        ) for d in docs]

    # all corpus records about one entity, oldest first, as a single compact card
    def entity_timeline(self, entity: str) -> list[Evidence]:
        docs = self._corpus.timeline(entity)
        if not docs:
            return [self._error(f"no corpus records for entity {entity}", "")]
        entries = []
        for d in docs[-settings.timeline_max_entries:]:
            meta = d.metadata
            date = str(meta.get("timestamp", ""))[:10]
            body = d.page_content.split("] ", 1)[-1][: settings.timeline_entry_char_cap]
            entries.append(f"{date} [{meta.get('record_id')}|rel {meta.get('reliability')}] {body}")
        snippet = f"timeline for {entity} ({len(docs)} records, oldest first): " + " || ".join(entries)
        return [self._store.register(source="corpus", ref=f"timeline:{entity.lower()}",
                                     snippet=snippet, snippet_cap=settings.tool_result_char_cap * 3)]

    # ---------- market ----------

    # fundamentals as 4 cards: core multiples, revenue trajectory, margin trend, cash/dilution
    def company_snapshot(self, ticker: str) -> list[Evidence]:
        ticker = ticker.upper()
        if not self._snap.ensure(ticker):
            return [self._error(f"no market data for {ticker}", "")]
        info = self._snap.info.get(ticker, {})
        stmts = self._snap.statements.get(ticker, {})
        cards = [self._card(ticker, "core", self._core_fields(ticker, info, stmts.get("income")))]
        for name, text in (("trajectory", self._revenue_trajectory(stmts.get("quarterly_income"))),
                           ("margins", self._margin_trend(stmts.get("income"))),
                           ("cash", self._cash_and_dilution(stmts.get("cashflow"), stmts.get("balance"), info))):
            if text:
                cards.append(self._card(ticker, name, text))
        return cards

    # register one snapshot aspect as its own citable evidence item
    def _card(self, ticker: str, aspect: str, text: str) -> Evidence:
        return self._store.register(source="yfinance", ref=f"snapshot:{ticker}:{aspect}",
                                    snippet=f"{ticker} {aspect}: {text}"[: settings.tool_result_char_cap],
                                    as_of=self._snap.as_of)

    # point-in-time valuation and profitability numbers
    @staticmethod
    def _core_fields(ticker: str, info: dict, income: pd.DataFrame | None) -> str:
        growth = None
        rev = _row(income, "Total Revenue")
        if rev is not None and len(rev) >= 2 and rev.iloc[1]:
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
            "trailing_eps": _fmt(info.get("trailingEps")),
            "free_cash_flow": _fmt(info.get("freeCashflow")),
        }
        return ", ".join(f"{k}={v}" for k, v in fields.items())

    # quarterly revenue with per-quarter YoY, so deceleration is visible
    @staticmethod
    def _revenue_trajectory(qincome: pd.DataFrame | None) -> str:
        rev = _row(qincome, "Total Revenue")
        if rev is None or len(rev) < 2:
            return ""
        rev = rev.iloc[: settings.quarters_shown]
        parts = []
        for i in range(len(rev)):
            label = str(rev.index[i])[:10]
            yoy = ""
            if i + 4 < len(rev) and rev.iloc[i + 4]:
                yoy = f" ({_fmt(float(rev.iloc[i] / rev.iloc[i + 4] - 1), pct=True)} YoY)"
            parts.append(f"{label}: {_fmt(float(rev.iloc[i]))}{yoy}")
        return "quarterly revenue, newest first: " + " | ".join(parts)

    # gross/operating/net margin by year, newest first
    @staticmethod
    def _margin_trend(income: pd.DataFrame | None) -> str:
        rev = _row(income, "Total Revenue")
        if rev is None:
            return ""
        parts = []
        for name, key in (("gross", "Gross Profit"), ("operating", "Operating Income"), ("net", "Net Income")):
            num = _row(income, key)
            if num is None:
                continue
            vals = []
            for i in range(min(settings.years_shown, len(num), len(rev))):
                if rev.iloc[i]:
                    vals.append(f"{str(num.index[i])[:4]}: {_fmt(float(num.iloc[i] / rev.iloc[i]), pct=True)}")
            if vals:
                parts.append(f"{name} margin by year (newest first): " + ", ".join(vals))
        return "; ".join(parts)

    # FCF history and share-count change
    @staticmethod
    def _cash_and_dilution(cashflow: pd.DataFrame | None, balance: pd.DataFrame | None, info: dict) -> str:
        parts = []
        fcf = _row(cashflow, "Free Cash Flow")
        if fcf is not None:
            vals = [f"{str(fcf.index[i])[:4]}: {_fmt(float(fcf.iloc[i]))}"
                    for i in range(min(settings.years_shown, len(fcf)))]
            parts.append("FCF by year (newest first): " + ", ".join(vals))
        shares = _row(balance, "Ordinary Shares Number")
        if shares is not None and len(shares) >= 2 and shares.iloc[-1]:
            change = float(shares.iloc[0] / shares.iloc[min(len(shares) - 1, settings.years_shown - 1)] - 1)
            parts.append(f"share count change over period: {_fmt(change, pct=True)}")
        if info.get("sharesOutstanding"):
            parts.append(f"shares outstanding: {_fmt(float(info['sharesOutstanding']))}")
        return "; ".join(parts)

    # two cards from full history: trend (returns, MAs, 52w range) and risk (vol, drawdown, volume, beta)
    def price_stats(self, ticker: str) -> list[Evidence]:
        ticker = ticker.upper()
        if not self._snap.ensure(ticker):
            return [self._error(f"no market data for {ticker}", "")]
        hist = self._snap.history[ticker]
        close = hist["Close"].dropna()
        window = close.tail(settings.drawdown_window_days)
        drawdown = float((window / window.cummax() - 1).min())
        daily = close.pct_change().dropna().tail(settings.vol_window_days)
        vol = float(daily.std() * np.sqrt(settings.trading_days_per_year))
        ma_s = float(close.tail(settings.ma_short).mean())
        ma_l = float(close.tail(settings.ma_long).mean())
        ma_l_std = float(close.tail(settings.ma_long).std())
        z = (float(close.iloc[-1]) - ma_l) / ma_l_std if ma_l_std else None
        range_pos = None
        year = close.tail(settings.trading_days_per_year)
        if len(year) > 1 and float(year.max()) != float(year.min()):
            range_pos = (float(close.iloc[-1]) - float(year.min())) / (float(year.max()) - float(year.min()))
        trend = {
            "last_close": _fmt(float(close.iloc[-1])),
            "return_6m": _fmt(_returns(close, settings.relative_strength_days), pct=True),
            "return_1y": _fmt(_returns(close, settings.trading_days_per_year), pct=True),
            f"vs_{settings.ma_short}dma": _fmt(float(close.iloc[-1]) / ma_s - 1, pct=True),
            f"vs_{settings.ma_long}dma": _fmt(float(close.iloc[-1]) / ma_l - 1, pct=True),
            f"zscore_vs_{settings.ma_long}dma": _fmt(z),
            "position_in_52w_range": _fmt(range_pos, pct=True),
            "relative_strength_6m_vs_index": _fmt(self._relative_strength(close)),
        }
        risk = {
            "vol_30d_annualized": _fmt(vol, pct=True),
            "max_drawdown_1y": _fmt(drawdown, pct=True),
            **self._volume_stats(hist),
            **self._index_risk(close),
        }
        return [
            self._store.register(source="yfinance", ref=f"price_stats:{ticker}:trend",
                                 snippet=(f"{ticker} price trend: " + ", ".join(f"{k}={v}" for k, v in trend.items()))[: settings.tool_result_char_cap],
                                 as_of=self._snap.as_of),
            self._store.register(source="yfinance", ref=f"price_stats:{ticker}:risk",
                                 snippet=(f"{ticker} risk & volume: " + ", ".join(f"{k}={v}" for k, v in risk.items()))[: settings.tool_result_char_cap],
                                 as_of=self._snap.as_of),
        ]

    # 6-month return minus the index's 6-month return
    def _relative_strength(self, close: pd.Series) -> float | None:
        bench = self._snap.history.get(settings.macro_tickers[0])
        if bench is None:
            return None
        r_t = _returns(close, settings.relative_strength_days)
        r_b = _returns(bench["Close"].dropna(), settings.relative_strength_days)
        return None if r_t is None or r_b is None else r_t - r_b

    # volume trend (30d vs 90d) and up-day vs down-day volume
    @staticmethod
    def _volume_stats(hist: pd.DataFrame) -> dict[str, str]:
        if "Volume" not in hist.columns:
            return {}
        vol = hist["Volume"].dropna()
        ret = hist["Close"].pct_change()
        recent, baseline = vol.tail(settings.volume_window_days), vol.tail(settings.volume_baseline_days)
        out = {}
        if len(baseline) and float(baseline.mean()):
            out["volume_30d_vs_90d"] = _fmt(float(recent.mean() / baseline.mean() - 1), pct=True)
        window = hist.tail(settings.volume_window_days)
        up = float(window.loc[ret.tail(settings.volume_window_days) > 0, "Volume"].sum())
        down = float(window.loc[ret.tail(settings.volume_window_days) < 0, "Volume"].sum())
        if down:
            out["upday_vs_downday_volume"] = _fmt(up / down)
        return out

    # 1-year correlation and beta vs the benchmark index
    def _index_risk(self, close: pd.Series) -> dict[str, str]:
        bench = self._snap.history.get(settings.macro_tickers[0])
        if bench is None:
            return {}
        joined = pd.concat([close.pct_change(), bench["Close"].pct_change()], axis=1, join="inner").dropna()
        joined = joined.tail(settings.trading_days_per_year)
        if len(joined) < settings.vol_window_days:
            return {}
        stock, idx = joined.iloc[:, 0], joined.iloc[:, 1]
        corr = float(stock.corr(idx))
        var = float(idx.var())
        beta = float(stock.cov(idx) / var) if var else None
        return {"correlation_1y_vs_index": _fmt(corr), "beta_1y_vs_index": _fmt(beta)}

    # valuation/growth table across the ticker and its peers
    def peer_compare(self, ticker: str, peers: list[str]) -> list[Evidence]:
        rows = []
        for t in [ticker, *peers][:6]:
            t = t.upper()
            if not self._snap.ensure(t):
                continue
            info = self._snap.info.get(t, {})
            rows.append(f"{t}: fPE={_fmt(info.get('forwardPE'))}, ev/ebitda={_fmt(info.get('enterpriseToEbitda'))}, "
                        f"rev_growth={_fmt(info.get('revenueGrowth'), pct=True)}, gm={_fmt(info.get('grossMargins'), pct=True)}, "
                        f"net_margin={_fmt(info.get('profitMargins'), pct=True)}")
        if not rows:
            return [self._error("no peer data", "")]
        snippet = "peer comparison: " + " | ".join(rows)
        return [self._store.register(source="yfinance", ref=f"peers:{ticker}:{','.join(peers)}",
                                     snippet=snippet[: settings.tool_result_char_cap], as_of=self._snap.as_of)]

    # index/rates/VIX levels and a computed risk_on / risk_off regime label
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

    # deterministic what-if: growth -> margin -> exit multiple -> implied price vs today
    def scenario_math(self, ticker: str, revenue_growth: float, net_margin: float,
                      exit_multiple: float) -> list[Evidence]:
        ticker = ticker.upper()
        if not self._snap.ensure(ticker):
            return [self._error(f"no market data for {ticker}", "")]
        info = self._snap.info.get(ticker, {})
        stmts = self._snap.statements.get(ticker, {})
        rev = _row(stmts.get("income"), "Total Revenue")
        revenue = float(rev.iloc[0]) if rev is not None else info.get("totalRevenue")
        shares = info.get("sharesOutstanding")
        close = self._snap.history[ticker]["Close"].dropna()
        price = float(close.iloc[-1])
        if not revenue or not shares:
            return [self._error(f"scenario needs revenue and share count for {ticker}", "")]
        projected_revenue = revenue * (1 + revenue_growth)
        implied_net_income = projected_revenue * net_margin
        implied_eps = implied_net_income / shares
        implied_price = implied_eps * exit_multiple
        move = implied_price / price - 1
        snippet = (f"{ticker} scenario: revenue {_fmt(revenue)} grows {_fmt(revenue_growth, pct=True)} "
                   f"-> {_fmt(projected_revenue)}; at net margin {_fmt(net_margin, pct=True)} "
                   f"net income {_fmt(implied_net_income)}, EPS {_fmt(implied_eps)}; "
                   f"at exit P/E {_fmt(exit_multiple)} implied price {_fmt(implied_price)} "
                   f"vs current {_fmt(price)} = {_fmt(move, pct=True)}")
        ref = f"scenario:{ticker}:g{revenue_growth}:m{net_margin}:x{exit_multiple}"
        return [self._store.register(source="yfinance", ref=ref,
                                     snippet=snippet[: settings.tool_result_char_cap], as_of=self._snap.as_of)]


# the tool list injected into every research-plan prompt
def tool_catalog() -> str:
    return json.dumps(TOOL_DESCRIPTIONS, indent=2)
