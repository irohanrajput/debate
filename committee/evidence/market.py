from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

from committee.config import settings


class MarketSnapshot:
    """Frozen market data: fetched eagerly at debate start, lazily filled for
    unforeseen tickers, then immutable. One truth for all agents, all rounds."""

    def __init__(self) -> None:
        self.history: dict[str, pd.DataFrame] = {}
        self.info: dict[str, dict] = {}
        self.statements: dict[str, dict[str, pd.DataFrame]] = {}
        self.as_of: str = datetime.now(timezone.utc).isoformat()
        self.offline: bool = False

    def fetch_eager(self, ticker: str | None) -> None:
        for t in settings.macro_tickers:
            self._fetch_history(t)
        if ticker:
            self.ensure(ticker)

    def ensure(self, ticker: str) -> bool:
        ticker = ticker.upper()
        if ticker in self.history:
            return True
        if self.offline:
            return False
        ok = self._fetch_history(ticker)
        if ok and not ticker.startswith("^"):
            self._fetch_fundamentals(ticker)
        return ok

    def _fetch_history(self, ticker: str) -> bool:
        try:
            df = yf.Ticker(ticker).history(period=settings.history_period, auto_adjust=True)
        except Exception:
            return False
        if df is None or df.empty:
            return False
        self.history[ticker] = df
        return True

    def _fetch_fundamentals(self, ticker: str) -> None:
        t = yf.Ticker(ticker)
        try:
            self.info[ticker] = dict(t.info or {})
        except Exception:
            self.info[ticker] = {}
        stmts: dict[str, pd.DataFrame] = {}
        for name, attr in (("income", "income_stmt"), ("balance", "balance_sheet"), ("cashflow", "cashflow")):
            try:
                df = getattr(t, attr)
                if df is not None and not df.empty:
                    stmts[name] = df
            except Exception:
                pass
        self.statements[ticker] = stmts

    def save(self, run_dir: Path) -> None:
        out = run_dir / "snapshot"
        out.mkdir(parents=True, exist_ok=True)
        for ticker, df in self.history.items():
            df.to_parquet(out / f"history_{ticker.replace('^', '_IDX_')}.parquet")
        (out / "info.json").write_text(json.dumps(self.info, default=str))
        (out / "meta.json").write_text(json.dumps({"as_of": self.as_of, "tickers": list(self.history)}))

    @classmethod
    def load(cls, run_dir: Path) -> "MarketSnapshot":
        snap = cls()
        src = run_dir / "snapshot"
        meta = json.loads((src / "meta.json").read_text())
        snap.as_of = meta["as_of"]
        for ticker in meta["tickers"]:
            snap.history[ticker] = pd.read_parquet(src / f"history_{ticker.replace('^', '_IDX_')}.parquet")
        snap.info = json.loads((src / "info.json").read_text())
        snap.offline = True
        return snap
