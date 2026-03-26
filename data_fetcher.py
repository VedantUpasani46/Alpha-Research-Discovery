"""
Data Fetcher
=============

Flexible, production-quality data acquisition layer with caching, rate
limiting, and validation.

Supported backends
------------------
* **Yahoo Finance** — equities, ETFs, indices via ``yfinance``.
* **Binance** — crypto spot markets via ``ccxt``.
* **Simulated / synthetic** — generate realistic price series for testing.

Usage
-----
.. code-block:: python

    fetcher = DataFetcher(source="yahoo", cache_dir="./cache")
    prices = fetcher.get_prices(["AAPL", "MSFT"], start="2020-01-01")
    returns = fetcher.get_returns(["AAPL", "MSFT"], start="2020-01-01")

All fetched data is automatically cached to disk (Parquet format) so that
repeated requests do not hit the API.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
import warnings
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".alpha_cache")


class DataSource(str, Enum):
    """Supported data backends."""

    YAHOO = "yahoo"
    BINANCE = "binance"
    SIMULATED = "simulated"


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


class _RateLimiter:
    """Simple token-bucket rate limiter.

    Parameters
    ----------
    calls_per_second : float
        Maximum calls per second.
    """

    def __init__(self, calls_per_second: float = 2.0) -> None:
        self._min_interval = 1.0 / calls_per_second
        self._last_call: float = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.monotonic()


# ---------------------------------------------------------------------------
# Data validation
# ---------------------------------------------------------------------------


class DataValidator:
    """Validates fetched price / volume data.

    Checks performed:
    * NaN fraction below threshold.
    * No duplicate indices.
    * No large gaps (configurable).
    * Outlier detection (returns > N sigma).
    """

    def __init__(
        self,
        max_nan_frac: float = 0.05,
        max_gap_days: int = 5,
        outlier_sigma: float = 10.0,
    ) -> None:
        self.max_nan_frac = max_nan_frac
        self.max_gap_days = max_gap_days
        self.outlier_sigma = outlier_sigma

    def validate(self, df: pd.DataFrame, name: str = "data") -> pd.DataFrame:
        """Run all validation checks.  Returns cleaned DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Data to validate.
        name : str
            Label for log messages.

        Returns
        -------
        pd.DataFrame
            Cleaned data (duplicates removed, NaN forward-filled up to a
            limit).
        """
        if df.empty:
            warnings.warn(f"[{name}] DataFrame is empty.")
            return df

        # 1. Duplicates
        n_dupes = df.index.duplicated().sum()
        if n_dupes > 0:
            logger.warning("[%s] Removing %d duplicate index entries.", name, n_dupes)
            df = df[~df.index.duplicated(keep="first")]

        # 2. Sort index
        df = df.sort_index()

        # 3. NaN fraction
        nan_frac = df.isna().mean()
        for col in nan_frac.index:
            if nan_frac[col] > self.max_nan_frac:
                logger.warning(
                    "[%s] Column '%s' has %.1f%% NaN (threshold %.1f%%).",
                    name, col, nan_frac[col] * 100, self.max_nan_frac * 100,
                )

        # 4. Forward-fill small gaps (up to max_gap_days)
        df = df.ffill(limit=self.max_gap_days)

        # 5. Gap detection (on DatetimeIndex only)
        if isinstance(df.index, pd.DatetimeIndex):
            diffs = df.index.to_series().diff()
            large_gaps = diffs[diffs > pd.Timedelta(days=self.max_gap_days)]
            if len(large_gaps) > 0:
                logger.warning(
                    "[%s] %d gaps > %d days detected. Largest: %s at %s.",
                    name,
                    len(large_gaps),
                    self.max_gap_days,
                    large_gaps.max(),
                    large_gaps.idxmax(),
                )

        # 6. Outlier detection on returns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            series = df[col].pct_change().dropna()
            if len(series) < 10:
                continue
            mu = series.mean()
            sigma = series.std()
            if sigma > 0:
                z_scores = ((series - mu) / sigma).abs()
                n_outliers = (z_scores > self.outlier_sigma).sum()
                if n_outliers > 0:
                    logger.warning(
                        "[%s] Column '%s': %d return outliers (|z| > %.1f).",
                        name, col, n_outliers, self.outlier_sigma,
                    )

        return df


# ---------------------------------------------------------------------------
# Cache layer
# ---------------------------------------------------------------------------


class _CacheManager:
    """Disk-based caching using Parquet files."""

    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, *args: Any) -> str:
        raw = "|".join(str(a) for a in args)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(self, *args: Any) -> Optional[pd.DataFrame]:
        key = self._cache_key(*args)
        path = self.cache_dir / f"{key}.parquet"
        if path.exists():
            logger.debug("Cache hit: %s", path)
            return pd.read_parquet(path)
        return None

    def put(self, df: pd.DataFrame, *args: Any) -> None:
        key = self._cache_key(*args)
        path = self.cache_dir / f"{key}.parquet"
        df.to_parquet(path)
        logger.debug("Cached to: %s", path)

    def clear(self) -> int:
        """Remove all cached files.  Returns count deleted."""
        count = 0
        for f in self.cache_dir.glob("*.parquet"):
            f.unlink()
            count += 1
        logger.info("Cleared %d cached files.", count)
        return count


# ---------------------------------------------------------------------------
# DataFetcher
# ---------------------------------------------------------------------------


class DataFetcher:
    """Unified data fetcher with caching, rate limiting, and validation.

    Parameters
    ----------
    source : DataSource or str
        Backend to use: ``"yahoo"``, ``"binance"``, or ``"simulated"``.
    cache_dir : str
        Directory for the Parquet cache.  Default ``~/.alpha_cache``.
    rate_limit : float
        Maximum API calls per second (default 2.0).
    validate : bool
        Run automatic data validation (default True).
    timezone_out : str
        Output timezone for timestamps (default ``"UTC"``).

    Examples
    --------
    >>> fetcher = DataFetcher("yahoo")
    >>> prices = fetcher.get_prices(["AAPL"], start="2023-01-01")
    >>> fetcher.clear_cache()
    """

    def __init__(
        self,
        source: Union[DataSource, str] = DataSource.YAHOO,
        cache_dir: str = _DEFAULT_CACHE_DIR,
        rate_limit: float = 2.0,
        validate: bool = True,
        timezone_out: str = "UTC",
    ) -> None:
        self.source = DataSource(source)
        self.cache = _CacheManager(cache_dir)
        self.limiter = _RateLimiter(rate_limit)
        self.validator = DataValidator() if validate else None
        self.timezone_out = timezone_out

    # -- Public API ----------------------------------------------------------

    def get_prices(
        self,
        symbols: Union[str, Sequence[str]],
        start: str = "2015-01-01",
        end: Optional[str] = None,
        interval: str = "1d",
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV price data.

        Parameters
        ----------
        symbols : str or list[str]
            Ticker(s) to fetch.
        start : str
            Start date (ISO format).
        end : str or None
            End date.  Defaults to today.
        interval : str
            Bar interval (e.g. ``"1d"``, ``"1h"``).
        columns : list[str] or None
            Subset of columns to return.  Default returns all OHLCV columns.

        Returns
        -------
        pd.DataFrame
            Multi-index (date, symbol) or simple DatetimeIndex if single
            symbol.  Columns: Open, High, Low, Close, Volume (at minimum).
        """
        symbols = [symbols] if isinstance(symbols, str) else list(symbols)
        end = end or datetime.now(timezone.utc).strftime("%Y-%m-%d")

        frames = []
        for sym in symbols:
            cached = self.cache.get("prices", self.source.value, sym, start, end, interval)
            if cached is not None:
                df = cached
            else:
                df = self._fetch_prices(sym, start, end, interval)
                if not df.empty:
                    self.cache.put(df, "prices", self.source.value, sym, start, end, interval)

            if self.validator and not df.empty:
                df = self.validator.validate(df, name=f"prices:{sym}")

            if not df.empty:
                df["symbol"] = sym
                frames.append(df)

        if not frames:
            logger.warning("No price data returned for %s", symbols)
            return pd.DataFrame()

        result = pd.concat(frames)

        # Timezone handling
        if isinstance(result.index, pd.DatetimeIndex):
            if result.index.tz is None:
                result.index = result.index.tz_localize("UTC")
            result.index = result.index.tz_convert(self.timezone_out)

        if columns:
            available = [c for c in columns if c in result.columns]
            result = result[available + ["symbol"]]

        if len(symbols) > 1:
            result = result.reset_index().set_index(["Date", "symbol"]).sort_index()

        return result

    def get_returns(
        self,
        symbols: Union[str, Sequence[str]],
        start: str = "2015-01-01",
        end: Optional[str] = None,
        method: str = "log",
    ) -> pd.DataFrame:
        """Fetch returns derived from closing prices.

        Parameters
        ----------
        symbols : str or list[str]
        start : str
        end : str or None
        method : str
            ``"log"`` for log returns, ``"simple"`` for arithmetic.

        Returns
        -------
        pd.DataFrame
            Returns with same structure as :meth:`get_prices`.
        """
        prices = self.get_prices(symbols, start, end, columns=["Close"])

        if "symbol" in prices.columns:
            # Multi-symbol: pivot, compute returns, unpivot
            pivot = prices.reset_index()
            if "Date" in pivot.columns and "symbol" in pivot.columns:
                wide = pivot.pivot(index="Date", columns="symbol", values="Close")
            else:
                wide = prices["Close"].unstack("symbol")

            if method == "log":
                ret = np.log(wide / wide.shift(1))
            else:
                ret = wide.pct_change()
            return ret.dropna(how="all")
        else:
            close = prices["Close"] if "Close" in prices.columns else prices.iloc[:, 0]
            if method == "log":
                ret = np.log(close / close.shift(1))
            else:
                ret = close.pct_change()
            return ret.dropna().to_frame("return")

    def get_volume(
        self,
        symbols: Union[str, Sequence[str]],
        start: str = "2015-01-01",
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch volume data.

        Returns
        -------
        pd.DataFrame
            Volume series.
        """
        return self.get_prices(symbols, start, end, columns=["Volume"])

    def get_options_data(
        self,
        symbol: str,
    ) -> pd.DataFrame:
        """Fetch options chain data (Yahoo Finance only).

        Parameters
        ----------
        symbol : str
            Underlying ticker.

        Returns
        -------
        pd.DataFrame
            Concatenated calls and puts with expiration dates.

        Raises
        ------
        NotImplementedError
            If source is not Yahoo Finance.
        """
        if self.source != DataSource.YAHOO:
            raise NotImplementedError(
                "Options data is only available from Yahoo Finance."
            )

        cached = self.cache.get("options", symbol)
        if cached is not None:
            return cached

        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("Install yfinance: pip install yfinance")

        self.limiter.wait()
        ticker = yf.Ticker(symbol)
        expirations = ticker.options

        frames = []
        for exp in expirations:
            self.limiter.wait()
            chain = ticker.option_chain(exp)
            calls = chain.calls.copy()
            calls["type"] = "call"
            calls["expiration"] = exp
            puts = chain.puts.copy()
            puts["type"] = "put"
            puts["expiration"] = exp
            frames.extend([calls, puts])

        if not frames:
            logger.warning("No options data for %s.", symbol)
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        self.cache.put(result, "options", symbol)
        logger.info("Fetched %d option contracts for %s.", len(result), symbol)
        return result

    def clear_cache(self) -> int:
        """Remove all cached data files.

        Returns
        -------
        int
            Number of files deleted.
        """
        return self.cache.clear()

    # -- Backend dispatchers -------------------------------------------------

    def _fetch_prices(
        self, symbol: str, start: str, end: str, interval: str
    ) -> pd.DataFrame:
        dispatch = {
            DataSource.YAHOO: self._fetch_yahoo,
            DataSource.BINANCE: self._fetch_binance,
            DataSource.SIMULATED: self._fetch_simulated,
        }
        return dispatch[self.source](symbol, start, end, interval)

    # -- Yahoo Finance -------------------------------------------------------

    def _fetch_yahoo(
        self, symbol: str, start: str, end: str, interval: str
    ) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError(
                "Yahoo Finance backend requires `yfinance`. "
                "Install with: pip install yfinance"
            )

        self.limiter.wait()
        logger.info("Fetching %s from Yahoo Finance (%s to %s).", symbol, start, end)

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end, interval=interval)
        except Exception as exc:
            logger.error("Yahoo Finance error for %s: %s", symbol, exc)
            return pd.DataFrame()

        if df.empty:
            logger.warning("No data returned for %s.", symbol)
            return pd.DataFrame()

        # Standardise column names
        df.columns = [c.title() for c in df.columns]
        # Keep core columns
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        return df[keep]

    # -- Binance (via ccxt) --------------------------------------------------

    def _fetch_binance(
        self, symbol: str, start: str, end: str, interval: str
    ) -> pd.DataFrame:
        try:
            import ccxt
        except ImportError:
            raise ImportError(
                "Binance backend requires `ccxt`. Install with: pip install ccxt"
            )

        exchange = ccxt.binance({"enableRateLimit": True})

        since = int(pd.Timestamp(start).timestamp() * 1000)
        end_ts = int(pd.Timestamp(end).timestamp() * 1000)

        # Map interval string
        tf_map = {"1d": "1d", "1h": "1h", "4h": "4h", "1w": "1w"}
        tf = tf_map.get(interval, "1d")

        logger.info("Fetching %s from Binance (%s to %s, tf=%s).", symbol, start, end, tf)

        all_ohlcv: List[list] = []
        limit = 1000

        while since < end_ts:
            self.limiter.wait()
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, tf, since=since, limit=limit)
            except Exception as exc:
                logger.error("Binance error for %s: %s", symbol, exc)
                break

            if not ohlcv:
                break

            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1  # next ms after last candle

        if not all_ohlcv:
            return pd.DataFrame()

        df = pd.DataFrame(
            all_ohlcv,
            columns=["timestamp", "Open", "High", "Low", "Close", "Volume"],
        )
        df["Date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("Date").drop(columns=["timestamp"])
        df = df[df.index <= pd.Timestamp(end, tz="UTC")]
        return df

    # -- Simulated / synthetic data ------------------------------------------

    def _fetch_simulated(
        self, symbol: str, start: str, end: str, interval: str
    ) -> pd.DataFrame:
        """Generate synthetic GBM price series for testing.

        Uses a Geometric Brownian Motion model with parameters derived from
        the symbol name hash so that each symbol produces a deterministic
        but distinct series.
        """
        logger.info("Generating simulated data for '%s'.", symbol)

        dates = pd.bdate_range(start=start, end=end)
        T = len(dates)
        if T == 0:
            return pd.DataFrame()

        # Deterministic seed from symbol name
        seed = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16) % (2 ** 31)
        rng = np.random.RandomState(seed)

        mu = 0.0002 + rng.uniform(-0.0001, 0.0003)  # daily drift
        sigma = 0.015 + rng.uniform(0, 0.01)  # daily vol
        S0 = 50 + rng.uniform(0, 200)

        # GBM
        dt = 1.0
        Z = rng.standard_normal(T)
        log_returns = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
        log_prices = np.log(S0) + np.cumsum(log_returns)
        close = np.exp(log_prices)

        # Synthetic OHLCV
        noise = rng.uniform(0.995, 1.005, size=(T, 3))
        open_ = close * noise[:, 0]
        high = np.maximum(close, open_) * (1 + rng.uniform(0, 0.01, T))
        low = np.minimum(close, open_) * (1 - rng.uniform(0, 0.01, T))
        volume = rng.lognormal(mean=15, sigma=0.5, size=T).astype(int)

        df = pd.DataFrame(
            {
                "Open": open_,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": volume,
            },
            index=dates,
        )
        df.index.name = "Date"
        return df
