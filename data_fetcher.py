"""
data_fetcher.py
───────────────
Unified data-access layer for the AI-Alpha-Factory.
Pulls from:
  • Yahoo Finance  (yfinance)         — daily OHLCV, macro ETFs
  • Binance REST   (python-binance)   — crypto OHLCV, funding rates, OI
  • CBOE / Deribit                    — options surfaces (stubs, extend as needed)

All data is cached locally as parquet files under ./cache/ to avoid redundant
network calls.  Every alpha module imports DataFetcher and nothing else.

Author : AI-Alpha-Factory
Version: 1.0.0
"""

from __future__ import annotations

import os
import hashlib
import logging
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("DataFetcher")

# ── optional imports (graceful degradation) ──────────────────────────────────
try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False
    log.warning("yfinance not installed — equity data unavailable")

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False
    log.warning("requests not installed — Binance REST unavailable")

CACHE_DIR = Path("./cache")
CACHE_DIR.mkdir(exist_ok=True)

# ── constants ─────────────────────────────────────────────────────────────────
BINANCE_BASE = "https://api.binance.com"
BINANCE_FAPI = "https://fapi.binance.com"   # futures

SP500_TICKERS: List[str] = [
    "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","BRK-B","JPM","V",
    "UNH","XOM","LLY","JNJ","WMT","MA","PG","HD","CVX","MRK","ABBV","KO",
    "AVGO","PEP","COST","ADBE","CSCO","MCD","TMO","ACN","NEE","DHR","ABT",
    "CRM","LIN","NKE","TXN","PM","ORCL","MS","BAC","AMGN","RTX","HON",
    "INTU","QCOM","AMD","SBUX","T","GS","CAT","BA","LOW","SPGI","AXP",
    "BLK","ISRG","DE","MDLZ","GILD","SYK","ADI","BKNG","REGN","PLD",
    "MMM","VRTX","CB","C","TJX","ZTS","MO","ADP","NOW","LRCX","ELV",
    "CI","SO","DUK","PNC","USB","TGT","EQIX","BSX","BDX","KLAC","MU",
    "ETN","NOC","ITW","GE","HUM","PANW","F","GM","CARR","FTNT",
]

CRYPTO_UNIVERSE: List[str] = [
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","ADAUSDT","XRPUSDT","DOGEUSDT",
    "AVAXUSDT","DOTUSDT","LINKUSDT","MATICUSDT","LTCUSDT","UNIUSDT","ATOMUSDT",
    "ETCUSDT","XLMUSDT","ALGOUSDT","VETUSDT","FILUSDT","TRXUSDT",
]


# ══════════════════════════════════════════════════════════════════════════════
class DataFetcher:
    """
    Central data-access class.  All expensive calls are cached to parquet.

    Usage
    -----
    df = DataFetcher()
    prices  = df.get_equity_prices(tickers, start, end)
    crypto  = df.get_crypto_ohlcv("BTCUSDT", "1d", start, end)
    funding = df.get_funding_rates("BTCUSDT", start, end)
    """

    def __init__(self, cache_dir: Path = CACHE_DIR, use_cache: bool = True):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = use_cache
        log.info("DataFetcher initialised | cache=%s", self.cache_dir)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _cache_path(self, key: str) -> Path:
        hashed = hashlib.md5(key.encode()).hexdigest()[:12]
        return self.cache_dir / f"{hashed}.parquet"

    def _load_cache(self, key: str) -> Optional[pd.DataFrame]:
        path = self._cache_path(key)
        if self.use_cache and path.exists():
            age_hours = (time.time() - path.stat().st_mtime) / 3600
            if age_hours < 24:
                log.debug("Cache hit  [%s] age=%.1fh", key[:40], age_hours)
                return pd.read_parquet(path)
        return None

    def _save_cache(self, key: str, df: pd.DataFrame) -> None:
        if not df.empty:
            df.to_parquet(self._cache_path(key))

    # ── equity data via yfinance ──────────────────────────────────────────────

    def get_equity_prices(
        self,
        tickers: List[str],
        start: str,
        end: str,
        field: str = "Adj Close",
    ) -> pd.DataFrame:
        """
        Returns a (date × ticker) DataFrame of adjusted close prices.
        Also caches raw OHLCV per ticker for volume-based alphas.
        """
        if not _YF_AVAILABLE:
            raise ImportError("yfinance required for equity data")

        cache_key = f"equity_prices|{'_'.join(sorted(tickers))}|{start}|{end}|{field}"
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        log.info("Downloading equity prices | tickers=%d | %s → %s", len(tickers), start, end)
        raw = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        else:
            prices = raw[["Close"]]
            prices.columns = tickers

        prices.index = pd.to_datetime(prices.index)
        prices.dropna(how="all", inplace=True)
        self._save_cache(cache_key, prices)
        return prices

    def get_equity_ohlcv(
        self,
        tickers: List[str],
        start: str,
        end: str,
    ) -> Dict[str, pd.DataFrame]:
        """
        Returns a dict  ticker → (date × [Open,High,Low,Close,Volume]) DataFrame.
        """
        if not _YF_AVAILABLE:
            raise ImportError("yfinance required")

        cache_key = f"equity_ohlcv|{'_'.join(sorted(tickers))}|{start}|{end}"
        cached = self._load_cache(cache_key)
        if cached is not None:
            # cached as wide frame; rebuild dict
            result: Dict[str, pd.DataFrame] = {}
            for t in tickers:
                cols = [c for c in cached.columns if c.endswith(f"_{t}")]
                if cols:
                    sub = cached[cols].copy()
                    sub.columns = [c.replace(f"_{t}", "") for c in sub.columns]
                    result[t] = sub
            return result

        log.info("Downloading equity OHLCV | tickers=%d", len(tickers))
        raw = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        result = {}
        if isinstance(raw.columns, pd.MultiIndex):
            for t in tickers:
                if t in raw.columns.get_level_values(1):
                    df = raw.xs(t, level=1, axis=1)[["Open","High","Low","Close","Volume"]]
                    df.index = pd.to_datetime(df.index)
                    result[t] = df.dropna()
        else:
            t = tickers[0]
            result[t] = raw[["Open","High","Low","Close","Volume"]].copy()

        # flatten for caching
        wide = pd.concat(
            {t: df.rename(columns={c: f"{c}_{t}" for c in df.columns}) for t, df in result.items()},
            axis=1,
        )
        wide.columns = wide.columns.droplevel(0)
        self._save_cache(cache_key, wide)
        return result

    # ── crypto OHLCV via Binance ──────────────────────────────────────────────

    def get_crypto_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """
        Returns (datetime × [Open,High,Low,Close,Volume]) for a Binance symbol.
        interval examples: "1m","5m","15m","1h","4h","1d"
        """
        if not _REQUESTS_AVAILABLE:
            raise ImportError("requests required for Binance data")

        cache_key = f"binance_ohlcv|{symbol}|{interval}|{start}|{end}"
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        log.info("Binance OHLCV | %s %s | %s → %s", symbol, interval, start, end)
        start_ms = int(pd.Timestamp(start).timestamp() * 1000)
        end_ms   = int(pd.Timestamp(end).timestamp()   * 1000)

        all_candles: List[list] = []
        limit = 1000
        current_ms = start_ms

        while current_ms < end_ms:
            url = f"{BINANCE_BASE}/api/v3/klines"
            params = {
                "symbol":    symbol,
                "interval":  interval,
                "startTime": current_ms,
                "endTime":   end_ms,
                "limit":     limit,
            }
            try:
                resp = requests.get(url, params=params, timeout=15)
                resp.raise_for_status()
                candles = resp.json()
            except Exception as exc:
                log.error("Binance request failed: %s", exc)
                break

            if not candles:
                break
            all_candles.extend(candles)
            current_ms = candles[-1][0] + 1
            if len(candles) < limit:
                break
            time.sleep(0.05)   # rate-limit courtesy

        if not all_candles:
            log.warning("No candles returned for %s %s", symbol, interval)
            return pd.DataFrame(columns=["Open","High","Low","Close","Volume"])

        df = pd.DataFrame(all_candles, columns=[
            "open_time","Open","High","Low","Close","Volume",
            "close_time","quote_volume","trades","taker_buy_base",
            "taker_buy_quote","ignore",
        ])
        df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df.set_index("datetime", inplace=True)
        df = df[["Open","High","Low","Close","Volume"]].astype(float)
        df = df[df.index < pd.Timestamp(end, tz="UTC")]
        self._save_cache(cache_key, df)
        return df

    def get_crypto_universe_daily(
        self,
        symbols: List[str],
        start: str,
        end: str,
    ) -> Dict[str, pd.DataFrame]:
        """Convenience wrapper — fetches daily OHLCV for a list of crypto symbols."""
        result: Dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                df = self.get_crypto_ohlcv(sym, "1d", start, end)
                if not df.empty:
                    result[sym] = df
            except Exception as exc:
                log.warning("Failed to fetch %s: %s", sym, exc)
        return result

    # ── funding rates (perpetuals) ────────────────────────────────────────────

    def get_funding_rates(
        self,
        symbol: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """
        Returns 8-hourly funding rates from Binance perpetual futures.
        Columns: [fundingRate, markPrice]
        """
        if not _REQUESTS_AVAILABLE:
            raise ImportError("requests required")

        cache_key = f"binance_funding|{symbol}|{start}|{end}"
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        log.info("Fetching funding rates | %s | %s → %s", symbol, start, end)
        start_ms = int(pd.Timestamp(start).timestamp() * 1000)
        end_ms   = int(pd.Timestamp(end).timestamp()   * 1000)

        all_rows: List[dict] = []
        current_ms = start_ms

        while current_ms < end_ms:
            url = f"{BINANCE_FAPI}/fapi/v1/fundingRate"
            params = {
                "symbol":    symbol,
                "startTime": current_ms,
                "endTime":   end_ms,
                "limit":     1000,
            }
            try:
                resp = requests.get(url, params=params, timeout=15)
                resp.raise_for_status()
                rows = resp.json()
            except Exception as exc:
                log.error("Funding rate request failed: %s", exc)
                break

            if not rows:
                break
            all_rows.extend(rows)
            current_ms = rows[-1]["fundingTime"] + 1
            if len(rows) < 1000:
                break
            time.sleep(0.05)

        if not all_rows:
            return pd.DataFrame(columns=["fundingRate","markPrice"])

        df = pd.DataFrame(all_rows)
        df["datetime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
        df.set_index("datetime", inplace=True)
        df["fundingRate"] = df["fundingRate"].astype(float)
        df["markPrice"]   = df["markPrice"].astype(float)
        df = df[["fundingRate","markPrice"]]
        self._save_cache(cache_key, df)
        return df

    # ── open interest ─────────────────────────────────────────────────────────

    def get_open_interest_history(
        self,
        symbol: str,
        period: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """
        Historical open interest from Binance futures.
        period: "5m","15m","30m","1h","2h","4h","6h","12h","1d"
        """
        if not _REQUESTS_AVAILABLE:
            raise ImportError("requests required")

        cache_key = f"binance_oi|{symbol}|{period}|{start}|{end}"
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        log.info("Fetching OI history | %s %s", symbol, period)
        start_ms = int(pd.Timestamp(start).timestamp() * 1000)
        end_ms   = int(pd.Timestamp(end).timestamp()   * 1000)

        all_rows: List[dict] = []
        current_ms = start_ms

        while current_ms < end_ms:
            url = f"{BINANCE_FAPI}/futures/data/openInterestHist"
            params = {
                "symbol":    symbol,
                "period":    period,
                "startTime": current_ms,
                "endTime":   end_ms,
                "limit":     500,
            }
            try:
                resp = requests.get(url, params=params, timeout=15)
                resp.raise_for_status()
                rows = resp.json()
            except Exception as exc:
                log.error("OI request failed: %s", exc)
                break

            if not rows or isinstance(rows, dict):
                break
            all_rows.extend(rows)
            current_ms = rows[-1]["timestamp"] + 1
            if len(rows) < 500:
                break
            time.sleep(0.05)

        if not all_rows:
            return pd.DataFrame(columns=["openInterest","sumOpenInterestValue"])

        df = pd.DataFrame(all_rows)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("datetime", inplace=True)
        df["openInterest"]     = df["sumOpenInterest"].astype(float)
        df["openInterestUSD"]  = df["sumOpenInterestValue"].astype(float)
        df = df[["openInterest","openInterestUSD"]]
        self._save_cache(cache_key, df)
        return df

    # ── macro / ETF proxies ───────────────────────────────────────────────────

    def get_macro_etfs(self, start: str, end: str) -> pd.DataFrame:
        """
        Fetches proxy ETFs for macro regime signals:
        SPY (equity), TLT (rates), GLD (gold), USO (oil), HYG (credit), UUP (USD).
        """
        etfs = ["SPY","TLT","GLD","USO","HYG","UUP","VIX"]
        try:
            prices = self.get_equity_prices(etfs, start, end)
        except Exception as exc:
            log.warning("Macro ETF fetch partial failure: %s", exc)
            prices = pd.DataFrame()
        return prices

    def get_vix(self, start: str, end: str) -> pd.Series:
        """Returns daily VIX close series."""
        try:
            raw = self.get_equity_prices(["^VIX"], start, end)
            return raw.iloc[:, 0].rename("VIX")
        except Exception:
            return pd.Series(dtype=float, name="VIX")

    # ── market-wide returns for beta computation ──────────────────────────────

    def get_market_returns(
        self,
        market_ticker: str = "SPY",
        start: str = "2019-01-01",
        end: str   = "2024-12-31",
    ) -> pd.Series:
        prices = self.get_equity_prices([market_ticker], start, end)
        col = prices.columns[0]
        return prices[col].pct_change().dropna().rename("market_return")


# ══════════════════════════════════════════════════════════════════════════════
#  Shared utility functions used by every alpha module
# ══════════════════════════════════════════════════════════════════════════════

def compute_returns(prices: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """Forward log-returns matrix (date × ticker)."""
    return np.log(prices / prices.shift(periods))


def cross_sectional_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize each row to [-1, +1] via rank then rescaling."""
    ranked = df.rank(axis=1, pct=True)   # [0, 1]
    return ranked.subtract(0.5).multiply(2)  # [-1, 1]


def winsorise(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    lb = series.quantile(lower)
    ub = series.quantile(upper)
    return series.clip(lb, ub)


def information_coefficient(
    signal: pd.Series,
    fwd_return: pd.Series,
    method: str = "spearman",
) -> float:
    """
    Spearman or Pearson IC between signal and forward return.
    Both series must share the same index.
    """
    joined = pd.concat([signal.rename("sig"), fwd_return.rename("fwd")], axis=1).dropna()
    if len(joined) < 5:
        return np.nan
    if method == "spearman":
        return joined["sig"].corr(joined["fwd"], method="spearman")
    return joined["sig"].corr(joined["fwd"])


def information_coefficient_matrix(
    signals: pd.DataFrame,
    forward_returns: pd.DataFrame,
    lags: List[int],
) -> pd.DataFrame:
    """
    Compute IC for each lag.
    signals/forward_returns: (date × ticker) DataFrames.
    Returns DataFrame (lag × stats): mean_IC, std_IC, ICIR, t_stat.
    """
    rows = []
    for lag in lags:
        fwd_lag = forward_returns.shift(-lag)
        ics = []
        for date in signals.index:
            if date not in fwd_lag.index:
                continue
            sig_row = signals.loc[date].dropna()
            fwd_row = fwd_lag.loc[date].dropna()
            common  = sig_row.index.intersection(fwd_row.index)
            if len(common) < 5:
                continue
            ic = information_coefficient(sig_row[common], fwd_row[common])
            ics.append(ic)
        ics_arr = np.array([x for x in ics if not np.isnan(x)])
        if len(ics_arr) < 3:
            rows.append({"lag": lag, "mean_IC": np.nan, "std_IC": np.nan, "ICIR": np.nan, "t_stat": np.nan, "n_obs": 0})
            continue
        mean_ic = ics_arr.mean()
        std_ic  = ics_arr.std(ddof=1)
        icir    = mean_ic / std_ic if std_ic > 0 else np.nan
        t_stat  = mean_ic / (std_ic / np.sqrt(len(ics_arr))) if std_ic > 0 else np.nan
        rows.append({"lag": lag, "mean_IC": mean_ic, "std_IC": std_ic, "ICIR": icir, "t_stat": t_stat, "n_obs": len(ics_arr)})
    return pd.DataFrame(rows).set_index("lag")


def compute_max_drawdown(pnl_series: pd.Series) -> float:
    """Maximum drawdown from cumulative PnL series."""
    cumulative = pnl_series.cumsum()
    rolling_max = cumulative.cummax()
    drawdown = cumulative - rolling_max
    return float(drawdown.min())


def compute_sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualised Sharpe ratio."""
    if returns.std() == 0 or returns.empty:
        return np.nan
    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))


def compute_turnover(signal_df: pd.DataFrame) -> float:
    """
    Average daily turnover as fraction of portfolio that changes.
    Normalised signal assumed to be in [-1, +1].
    """
    delta = signal_df.diff().abs().sum(axis=1)
    total_long = signal_df.clip(lower=0).sum(axis=1) * 2   # denominator
    turnover = (delta / total_long.replace(0, np.nan)).dropna()
    return float(turnover.mean())


def long_short_portfolio_returns(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    top_pct: float = 0.20,
    transaction_cost_bps: float = 5.0,
) -> pd.Series:
    """
    Each day: long top `top_pct` of signal, short bottom `top_pct`.
    Returns daily PnL series (long-short, net of transaction costs).
    """
    port_returns = []
    prev_weights: Optional[pd.Series] = None

    for date in signal.index:
        if date not in returns.index:
            continue
        sig_row  = signal.loc[date].dropna()
        ret_row  = returns.loc[date].dropna()
        common   = sig_row.index.intersection(ret_row.index)
        if len(common) < 10:
            port_returns.append((date, np.nan))
            continue

        sig   = sig_row[common]
        rets  = ret_row[common]
        n     = len(common)
        n_top = max(1, int(n * top_pct))

        long_assets  = sig.nlargest(n_top).index
        short_assets = sig.nsmallest(n_top).index

        weights = pd.Series(0.0, index=common)
        weights[long_assets]  =  1.0 / n_top
        weights[short_assets] = -1.0 / n_top

        gross_return = (weights * rets).sum()

        # transaction cost
        if prev_weights is not None:
            prev_w = prev_weights.reindex(common).fillna(0)
            turnover = (weights - prev_w).abs().sum() / 2.0
            cost = turnover * transaction_cost_bps / 10_000
        else:
            cost = transaction_cost_bps / 10_000

        net_return = gross_return - cost
        port_returns.append((date, net_return))
        prev_weights = weights

    return pd.Series(
        {d: v for d, v in port_returns if not np.isnan(v)},
        name="port_return",
    )


def fama_macbeth_regression(
    signal: pd.DataFrame,
    forward_returns: pd.DataFrame,
    lag: int = 1,
) -> Dict[str, float]:
    """
    Fama-MacBeth: run cross-sectional regression each date, return
    time-series mean and t-stat of the slope coefficient.
    """
    from scipy import stats as sp_stats

    fwd = forward_returns.shift(-lag)
    slopes, intercepts = [], []

    for date in signal.index:
        if date not in fwd.index:
            continue
        x = signal.loc[date].dropna()
        y = fwd.loc[date].dropna()
        common = x.index.intersection(y.index)
        if len(common) < 5:
            continue
        slope, intercept, *_ = sp_stats.linregress(x[common].values, y[common].values)
        slopes.append(slope)
        intercepts.append(intercept)

    if len(slopes) < 5:
        return {"gamma": np.nan, "t_stat": np.nan, "n_periods": 0}

    arr = np.array(slopes)
    t_stat = arr.mean() / (arr.std(ddof=1) / np.sqrt(len(arr)))
    return {
        "gamma":     float(arr.mean()),
        "t_stat":    float(t_stat),
        "n_periods": len(arr),
    }


def walk_forward_split(
    index: pd.DatetimeIndex,
    is_frac: float = 0.70,
) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Simple expanding IS/OOS split."""
    split_idx = int(len(index) * is_frac)
    return index[:split_idx], index[split_idx:]
