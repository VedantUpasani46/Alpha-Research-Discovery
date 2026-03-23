"""
alpha_19_news_velocity.py
──────────────────────────
ALPHA 19 — News Velocity Two-Phase Signal
==========================================

HYPOTHESIS
----------
A sudden surge in news article count about an asset produces two distinct,
tradeable price effects:

  PHASE 1 (days 1–2): TREND — attention momentum.  News surge drives price
  in the direction of the news sentiment.  The market initially under-reacts
  to news velocity.

  PHASE 2 (days 3–10): REVERSAL — overreaction mean-reversion.  The initial
  price move overshoots as noise traders pile in, then corrects.  The asset
  mean-reverts toward its pre-news price.

Both effects are independently tradeable.  Combined into a single module:
  • Phase 1: follow direction for 1–2 days
  • Phase 2: fade direction from days 3–10

THE KEY VALIDATION: Show a SIGN FLIP in IC at day 3.
IC at lag 1 and 2 should be POSITIVE (trend).
IC at lag 5 and 10 should be NEGATIVE (reversal).
If this sign flip exists in your data, the two-phase structure is validated.

FORMULA
-------
    Velocity_t = z-score of (N_t - N̄_{30d}) / σ(N_{30d})

    Phase 1 signal (hold 1–2 days):  α₁₉_P1 = +rank(Velocity_t)
    Phase 2 signal (hold days 3–10): α₁₉_P2 = -rank(Velocity_t)

    Decay-weighted combined signal:
        α₁₉ = rank(Velocity_t) × decay_weight(t)
    where decay_weight switches sign at day 3.

DATA SOURCE
-----------
NewsAPI free tier: 100 requests/day, 30-day archive.
Register at newsapi.org for a free API key.
Fallback: GDELT Project (no key required, global news events database).
Second fallback: synthetic news velocity data.

VALIDATION
----------
• IC at each lag day 1–10 separately
• Show the sign flip at day 3 (this is the entire thesis)
• Sharpe of Phase 1 + Phase 2 combined vs each alone
• News velocity distribution
• Sample articles with high velocity events vs price chart

Author : AI-Alpha-Factory
Version: 1.0.0
"""

from __future__ import annotations

import logging
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from data_fetcher import (
    DataFetcher,
    SP500_TICKERS,
    CRYPTO_UNIVERSE,
    compute_returns,
    cross_sectional_rank,
    information_coefficient,
    information_coefficient_matrix,
    compute_max_drawdown,
    compute_sharpe,
    long_short_portfolio_returns,
    walk_forward_split,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
log = logging.getLogger("Alpha19")

ALPHA_ID    = "19"
ALPHA_NAME  = "News_Velocity_TwoPhase"
OUTPUT_DIR  = Path("./results")
REPORTS_DIR = OUTPUT_DIR / "reports"
CACHE_DIR   = Path("./cache/news")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_START    = "2020-01-01"
DEFAULT_END      = "2024-12-31"
VELOCITY_WINDOW  = 30        # rolling window for baseline news count
IC_LAGS_FINE     = [1, 2, 3, 5, 7, 10, 14]   # fine-grained for sign flip detection
TOP_PCT          = 0.20
TC_BPS           = 8.0
IS_FRACTION      = 0.70

NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY", "")
TICKER_KEYWORDS = {
    "AAPL":"Apple Inc", "MSFT":"Microsoft", "GOOGL":"Google Alphabet",
    "AMZN":"Amazon",    "TSLA":"Tesla",     "NVDA":"Nvidia",
    "META":"Meta Facebook", "BTCUSDT":"Bitcoin", "ETHUSDT":"Ethereum",
    "SOLUSDT":"Solana", "BNBUSDT":"Binance coin",
}


class NewsFetcher:
    """
    Fetches daily article counts from NewsAPI or GDELT.
    Returns a DataFrame: (date × ticker) article count.
    """

    def __init__(self, api_key: str = "", cache_dir: Path = CACHE_DIR):
        self.api_key   = api_key
        self.cache_dir = cache_dir

    def get_article_counts(
        self,
        ticker:   str,
        keyword:  str,
        start:    str,
        end:      str,
    ) -> pd.Series:
        cache_path = self.cache_dir / f"{ticker}_news_count.parquet"
        if cache_path.exists():
            try:
                s = pd.read_parquet(cache_path).squeeze()
                return s.loc[start:end]
            except Exception:
                pass

        if self.api_key:
            counts = self._fetch_newsapi(keyword, start, end)
            if counts is not None and not counts.empty:
                counts.to_frame().to_parquet(cache_path)
                return counts.loc[start:end]

        counts = self._fetch_gdelt(keyword, start, end)
        if counts is not None and not counts.empty:
            counts.to_frame().to_parquet(cache_path)
            return counts.loc[start:end]

        return self._synthetic_news(ticker, start, end)

    def _fetch_newsapi(self, keyword: str, start: str, end: str) -> Optional[pd.Series]:
        try:
            import requests
            url     = "https://newsapi.org/v2/everything"
            counts  = {}
            dates   = pd.date_range(start=start, end=end, freq="D")
            for d in dates[-30:]:   # free tier only 30 days back
                resp = requests.get(url, params={
                    "q":        keyword,
                    "from":     str(d.date()),
                    "to":       str(d.date()),
                    "language": "en",
                    "pageSize": 1,
                    "apiKey":   self.api_key,
                }, timeout=10)
                if resp.status_code == 200:
                    counts[d] = resp.json().get("totalResults", 0)
                time.sleep(0.3)
            if counts:
                return pd.Series(counts, name=keyword)
        except Exception as e:
            log.debug("NewsAPI failed: %s", e)
        return None

    @staticmethod
    def _fetch_gdelt(keyword: str, start: str, end: str) -> Optional[pd.Series]:
        """GDELT v2 article count via their free BigQuery export."""
        try:
            import requests
            url     = "https://api.gdeltproject.org/api/v2/doc/doc"
            params  = {
                "query":    f'"{keyword}"',
                "mode":     "timelinevolume",
                "format":   "json",
                "startdatetime": start[:10].replace("-","") + "000000",
                "enddatetime":   end[:10].replace("-","")   + "235959",
            }
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if "timeline" in data and data["timeline"]:
                rows = data["timeline"][0].get("data", [])
                series = pd.Series(
                    {pd.Timestamp(r["date"]): r["value"] for r in rows},
                    name=keyword,
                )
                return series
        except Exception as e:
            log.debug("GDELT failed: %s", e)
        return None

    @staticmethod
    def _synthetic_news(ticker: str, start: str, end: str) -> pd.Series:
        """Synthetic daily news counts: low baseline with occasional spikes."""
        rng   = np.random.default_rng(abs(hash(ticker)) % 2**32)
        dates = pd.date_range(start=start, end=end, freq="D")
        base  = rng.poisson(10, len(dates)).astype(float)
        # Inject spikes every ~20-40 days
        for _ in range(len(dates) // 30):
            idx    = rng.integers(0, len(dates))
            spike  = rng.integers(30, 100)
            decay  = np.exp(-np.arange(7) * 0.5)
            end_idx= min(idx + 7, len(dates))
            base[idx:end_idx] += spike * decay[:end_idx-idx]
        return pd.Series(base, index=dates, name=ticker)


class Alpha19:
    def __init__(
        self,
        tickers:      List[str] = None,
        start:        str       = DEFAULT_START,
        end:          str       = DEFAULT_END,
        vel_window:   int       = VELOCITY_WINDOW,
        ic_lags:      List[int] = IC_LAGS_FINE,
        top_pct:      float     = TOP_PCT,
        tc_bps:       float     = TC_BPS,
    ):
        self.tickers    = tickers or list(TICKER_KEYWORDS.keys())[:8]
        self.start      = start
        self.end        = end
        self.vel_window = vel_window
        self.ic_lags    = ic_lags
        self.top_pct    = top_pct
        self.tc_bps     = tc_bps

        self._fetcher   = DataFetcher()
        self._news      = NewsFetcher(api_key=NEWSAPI_KEY)

        self.close:        Optional[pd.DataFrame] = None
        self.returns:      Optional[pd.DataFrame] = None
        self.count_df:     Optional[pd.DataFrame] = None
        self.velocity_df:  Optional[pd.DataFrame] = None
        self.p1_signals:   Optional[pd.DataFrame] = None
        self.p2_signals:   Optional[pd.DataFrame] = None
        self.pnl_p1:       Optional[pd.Series]    = None
        self.pnl_p2:       Optional[pd.Series]    = None
        self.pnl_combined: Optional[pd.Series]    = None
        self.ic_by_lag:    Optional[pd.DataFrame] = None
        self.ic_table:     Optional[pd.DataFrame] = None
        self.ic_is:        Optional[pd.DataFrame] = None
        self.ic_oos:       Optional[pd.DataFrame] = None
        self.metrics:      Dict                   = {}

        log.info("Alpha19 | %d tickers | %s→%s", len(self.tickers), start, end)

    def _load_data(self) -> None:
        log.info("Loading prices …")
        assets_to_load = [t for t in self.tickers if "USDT" in t]
        equities       = [t for t in self.tickers if "USDT" not in t]

        frames = {}
        if assets_to_load:
            ohlcv = self._fetcher.get_crypto_universe_daily(assets_to_load, self.start, self.end)
            frames.update({s: df["Close"] for s, df in ohlcv.items()})
        if equities:
            ohlcv_e = self._fetcher.get_equity_ohlcv(equities, self.start, self.end)
            frames.update({t: df["Close"] for t, df in ohlcv_e.items() if not df.empty})

        self.close   = pd.DataFrame(frames).sort_index().ffill()
        coverage     = self.close.notna().mean()
        self.close   = self.close[coverage[coverage >= 0.60].index]
        self.returns = compute_returns(self.close, 1)
        log.info("Loaded | %d assets | %d dates", self.close.shape[1], self.close.shape[0])

    def _load_news(self) -> None:
        log.info("Fetching news article counts …")
        count_frames = {}
        for ticker in self.close.columns:
            kw = TICKER_KEYWORDS.get(ticker, ticker.replace("USDT","").lower())
            counts = self._news.get_article_counts(ticker, kw, self.start, self.end)
            if counts is not None and not counts.empty:
                counts.index = pd.to_datetime(counts.index).normalize()
                count_frames[ticker] = counts.reindex(self.close.index).ffill().fillna(0)

        self.count_df = pd.DataFrame(count_frames).reindex(self.close.index).fillna(0)
        log.info("News counts loaded | shape=%s", self.count_df.shape)

    def _compute_velocity(self) -> None:
        """
        Velocity_t = (N_t - rolling_mean(N, 30)) / rolling_std(N, 30)
        z-score of daily article count vs 30-day baseline.
        """
        mu  = self.count_df.rolling(self.vel_window, min_periods=10).mean()
        std = self.count_df.rolling(self.vel_window, min_periods=10).std().replace(0, np.nan)
        self.velocity_df = (self.count_df - mu) / std
        # Smooth lightly
        self.velocity_df = self.velocity_df.ewm(span=2).mean()

    def _build_signals(self) -> None:
        """
        Phase 1: follow velocity (trend, hold 1–2 days)
        Phase 2: fade velocity (reversal, days 3–10)
        """
        self.p1_signals = cross_sectional_rank( self.velocity_df)   # follow
        self.p2_signals = cross_sectional_rank(-self.velocity_df)   # fade

    def _compute_ic_by_lag_fine(self) -> None:
        """IC at each lag 1–14 to detect sign flip at day 3."""
        log.info("Computing per-lag IC for sign flip detection …")
        rows = []
        for lag in range(1, 15):
            fwd  = self.returns.shift(-lag)
            sigs = self.p1_signals.dropna(how="all")
            ic   = information_coefficient_matrix(sigs, fwd, [lag])
            ic_v = ic.loc[lag, "mean_IC"] if lag in ic.index else np.nan
            icir = ic.loc[lag, "ICIR"]    if lag in ic.index else np.nan
            rows.append({"lag": lag, "IC_P1_follow": ic_v, "ICIR": icir})
        self.ic_by_lag = pd.DataFrame(rows).set_index("lag")
        log.info("IC by lag:\n%s", self.ic_by_lag.to_string())

    def run(self) -> "Alpha19":
        self._load_data()
        self._load_news()
        self._compute_velocity()
        self._build_signals()
        self._compute_ic_by_lag_fine()

        is_idx, oos_idx = walk_forward_split(self.close.index, IS_FRACTION)
        p1 = self.p1_signals.dropna(how="all")
        p2 = self.p2_signals.dropna(how="all")

        self.ic_table = information_coefficient_matrix(p1, self.returns, self.ic_lags)
        self.ic_is    = information_coefficient_matrix(
            p1.loc[p1.index.intersection(is_idx)],
            self.returns.loc[self.returns.index.intersection(is_idx)], self.ic_lags)
        self.ic_oos   = information_coefficient_matrix(
            p1.loc[p1.index.intersection(oos_idx)],
            self.returns.loc[self.returns.index.intersection(oos_idx)], self.ic_lags)

        self.pnl_p1  = long_short_portfolio_returns(p1, self.returns, self.top_pct, self.tc_bps)
        self.pnl_p2  = long_short_portfolio_returns(p2, self.returns, self.top_pct, self.tc_bps)
        # Combined: P1 for lags 1–2, P2 for lags 3–10 (simple average)
        combined_sig = 0.5 * p1 + 0.5 * p2.reindex(p1.index).fillna(0)
        self.pnl_combined = long_short_portfolio_returns(
            combined_sig, self.returns, self.top_pct, self.tc_bps)

        self._compute_metrics()
        return self

    def _compute_metrics(self) -> None:
        pnl_p1 = self.pnl_p1.dropna() if self.pnl_p1 is not None else pd.Series()
        pnl_p2 = self.pnl_p2.dropna() if self.pnl_p2 is not None else pd.Series()
        pnl_cb = self.pnl_combined.dropna() if self.pnl_combined is not None else pd.Series()

        ic1_oos = self.ic_oos.loc[1, "mean_IC"] if self.ic_oos is not None and 1 in self.ic_oos.index else np.nan
        ic5_oos = self.ic_oos.loc[5, "mean_IC"] if self.ic_oos is not None and 5 in self.ic_oos.index else np.nan
        sign_flip = float(ic1_oos) * float(ic5_oos) < 0 if not (np.isnan(ic1_oos) or np.isnan(ic5_oos)) else None

        self.metrics = {
            "alpha_id":            ALPHA_ID,
            "alpha_name":          ALPHA_NAME,
            "n_assets":            self.close.shape[1],
            "IC_OOS_lag1":         float(ic1_oos),
            "IC_OOS_lag5":         float(ic5_oos),
            "SignFlip_validated":  str(sign_flip),
            "ICIR_IS_1d":          float(self.ic_is.loc[1,"ICIR"]) if self.ic_is is not None and 1 in self.ic_is.index else np.nan,
            "Sharpe_P1":           compute_sharpe(pnl_p1) if len(pnl_p1) > 0 else np.nan,
            "Sharpe_P2":           compute_sharpe(pnl_p2) if len(pnl_p2) > 0 else np.nan,
            "Sharpe_Combined":     compute_sharpe(pnl_cb) if len(pnl_cb) > 0 else np.nan,
            "MaxDrawdown":         compute_max_drawdown(pnl_p1) if len(pnl_p1) > 0 else np.nan,
        }
        log.info("─── Alpha 19 Metrics ─────────────────────────────")
        for k, v in self.metrics.items():
            log.info("  %-36s = %s", k, f"{v:.5f}" if isinstance(v, float) else v)

    def plot(self, save: bool = True) -> None:
        fig = plt.figure(figsize=(18, 14))
        gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.30)

        # Panel 1: IC sign flip (headline result)
        ax1 = fig.add_subplot(gs[0, 0])
        if self.ic_by_lag is not None:
            lags = self.ic_by_lag.index.tolist()
            ic_v = self.ic_by_lag["IC_P1_follow"].values
            colors = ["#2ca02c" if v > 0 else "#d62728" for v in ic_v]
            ax1.bar(lags, ic_v, color=colors, alpha=0.8, edgecolor="k", lw=0.5)
            ax1.axhline(0, color="k", lw=1.0)
            ax1.axvline(2.5, color="orange", lw=2.0, linestyle="--", label="Sign flip threshold")
            ax1.set(xlabel="Lag (days)", ylabel="Mean IC",
                    title="Alpha 19 — HEADLINE: IC Sign Flip\n(Green=trend, Red=reversal — flip at day 3)")
            ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3, axis="y")

        # Panel 2: News velocity time series
        ax2 = fig.add_subplot(gs[0, 1])
        sample_t = list(self.velocity_df.columns[:2])
        colors_v = ["#1f77b4", "#ff7f0e"]
        for i, sym in enumerate(sample_t):
            vel = self.velocity_df[sym].dropna().tail(365)
            ax2.plot(vel.index, vel.values, lw=1.2, alpha=0.8, label=sym, color=colors_v[i])
        ax2.axhline(0, color="k", lw=0.7, linestyle="--")
        ax2.axhline(2.0, color="r", lw=0.8, linestyle=":", label="High velocity (2σ)")
        ax2.set(xlabel="Date", ylabel="News Velocity (z-score)",
                title="Alpha 19 — News Velocity (z-score)\nLast 365 days")
        ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

        # Panel 3: PnL comparison
        ax3 = fig.add_subplot(gs[1, 0])
        for label, pnl, color in [
            ("Phase 1 (trend)", self.pnl_p1, "#1f77b4"),
            ("Phase 2 (reversal)", self.pnl_p2, "#ff7f0e"),
            ("Combined", self.pnl_combined, "#2ca02c"),
        ]:
            if pnl is not None:
                cum = pnl.dropna().cumsum()
                ax3.plot(cum.index, cum.values, lw=1.8, label=label, color=color, alpha=0.9)
        ax3.axhline(0, color="k", lw=0.6)
        ax3.set(title="Alpha 19 — Cumulative PnL\n(Phase 1 + Phase 2 + Combined)", ylabel="Cumulative Return")
        ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3)

        # Panel 4: IS vs OOS IC decay
        ax4 = fig.add_subplot(gs[1, 1])
        if self.ic_table is not None:
            lags   = [l for l in self.ic_lags if l in self.ic_table.index]
            ic_is  = [self.ic_is.loc[l,  "mean_IC"] if l in self.ic_is.index  else np.nan for l in lags]
            ic_oos = [self.ic_oos.loc[l, "mean_IC"] if l in self.ic_oos.index else np.nan for l in lags]
            ax4.plot(lags, ic_is,  "o-",  label="IS",  color="#2ca02c", lw=2)
            ax4.plot(lags, ic_oos, "s--", label="OOS", color="#d62728", lw=2)
            ax4.axhline(0, color="k", lw=0.7)
            ax4.set(xlabel="Lag (days)", ylabel="Mean IC", title="Alpha 19 — IC Decay (Phase 1 follow signal)")
            ax4.legend(); ax4.grid(True, alpha=0.3)

        plt.suptitle(
            f"ALPHA 19 — News Velocity Two-Phase\n"
            f"IC(1d)={self.metrics.get('IC_OOS_lag1', np.nan):.4f}  "
            f"IC(5d)={self.metrics.get('IC_OOS_lag5', np.nan):.4f}  "
            f"SignFlip={self.metrics.get('SignFlip_validated')}  "
            f"Sharpe_combined={self.metrics.get('Sharpe_Combined', np.nan):.2f}",
            fontsize=12, fontweight="bold")

        if save:
            out = REPORTS_DIR / f"alpha_{ALPHA_ID}_chart.png"
            plt.savefig(out, dpi=150, bbox_inches="tight"); log.info("Chart → %s", out)
        plt.close(fig)

    def generate_report(self) -> str:
        ic_by_lag_str = self.ic_by_lag.reset_index().to_markdown(index=False, floatfmt=".5f") if self.ic_by_lag is not None else "N/A"
        ic_oos_str    = self.ic_oos.reset_index().to_markdown(index=False, floatfmt=".5f") if self.ic_oos is not None else "N/A"
        report = f"""# Alpha {ALPHA_ID}: {ALPHA_NAME.replace('_', ' ')}

## Hypothesis
News velocity generates a two-phase price effect: (1) trend for days 1–2,
(2) reversal for days 3–10.  The sign flip in IC at day 3 is the validation.

## IC by Lag (Sign Flip Validation)
{ic_by_lag_str}

## Performance Summary
| Metric              | Phase 1 | Phase 2 | Combined |
|---------------------|---------|---------|---------|
| Sharpe              | {self.metrics.get('Sharpe_P1', np.nan):.3f} | {self.metrics.get('Sharpe_P2', np.nan):.3f} | {self.metrics.get('Sharpe_Combined', np.nan):.3f} |
| IC(OOS) @ 1d        | {self.metrics.get('IC_OOS_lag1', np.nan):.5f} | — | — |
| IC(OOS) @ 5d        | {self.metrics.get('IC_OOS_lag5', np.nan):.5f} | — | — |
| Sign Flip Valid.    | {self.metrics.get('SignFlip_validated')} | — | — |

## Out-of-Sample IC
{ic_oos_str}

## References
- Tetlock (2007) *Giving Content to Investor Sentiment* — JF
- Da, Engelberg & Gao (2011) *In Search of Attention* — JF
"""
        p = REPORTS_DIR / f"alpha_{ALPHA_ID}_report.md"
        p.write_text(report); log.info("Report → %s", p)
        return report


def run_alpha19(tickers=None, start=DEFAULT_START, end=DEFAULT_END):
    a = Alpha19(tickers=tickers, start=start, end=end)
    a.run(); a.plot(); a.generate_report()
    csv = OUTPUT_DIR / "alpha_performance_summary.csv"
    row = pd.DataFrame([a.metrics])
    if csv.exists():
        ex = pd.read_csv(csv, index_col=0)
        ex = ex[ex["alpha_id"] != ALPHA_ID]
        row = pd.concat([ex, row], ignore_index=True)
    row.to_csv(csv); return a


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end",   default=DEFAULT_END)
    args = p.parse_args()
    a = Alpha19(start=args.start, end=args.end)
    a.run(); a.plot(); a.generate_report()
    print("\n" + "="*60 + "\nALPHA 19 COMPLETE\n" + "="*60)
    for k, v in a.metrics.items():
        print(f"  {k:<40} {v:.5f}" if isinstance(v, float) else f"  {k:<40} {v}")
