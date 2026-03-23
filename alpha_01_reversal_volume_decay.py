"""
alpha_01_reversal_volume_decay.py
──────────────────────────────────
ALPHA 01 — Cross-Sectional Short-Term Reversal with Volume Decay
================================================================

HYPOTHESIS
----------
Assets with the largest 1-day losses in a cross-section tend to mean-revert
in the short term (classic DeBondt-Thaler short-term reversal).  However,
when volume is abnormally HIGH during the loss, the move is more likely
driven by informed order flow rather than noise trading — and therefore the
reversal is weaker.  Volume-adjusting the reversal signal dampens it in
high-volume environments, improving out-of-sample IC.

FORMULA
-------
    α₁ = -rank(r_{t-1}) × exp(-λ × V_{t-1} / V̄₂₀)

where:
    r_{t-1}  = 1-day lagged return
    V_{t-1}  = yesterday's volume
    V̄₂₀     = 20-day rolling average volume  (normalisation denominator)
    λ        = decay parameter tuned by grid search over [0.5, 2.0]
               higher λ → stronger discounting of high-volume days

ASSET CLASS
-----------
Equity cross-section (S&P 500) **or** Crypto top-50 basket (Binance).
This module defaults to the S&P 500 equity universe for maximum depth.

REBALANCE FREQUENCY
-------------------
Daily.  Long top-20% by α₁, short bottom-20%.

VALIDATION
----------
• IC at lag 1, 2, 3, 5, 10 days (in-sample and out-of-sample)
• Compare IC to naive reversal (without volume adjustment)  → show lift
• Fama-MacBeth t-statistic target > 2.0
• Kupiec proportion-of-failures backtest on tail risk
• Sharpe ratio, Max Drawdown, Annualised Turnover

REFERENCES
----------
• Jegadeesh (1990) — short-term return reversals
• Avramov, Chordia & Goyal (2006) — trading activity and expected returns
• Lehmann (1990) — fads, martingales and market efficiency

Author : AI-Alpha-Factory
Version: 1.0.0
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats as sp_stats

from data_fetcher import (
    DataFetcher,
    SP500_TICKERS,
    CRYPTO_UNIVERSE,
    compute_returns,
    cross_sectional_rank,
    winsorise,
    information_coefficient,
    information_coefficient_matrix,
    compute_max_drawdown,
    compute_sharpe,
    compute_turnover,
    long_short_portfolio_returns,
    fama_macbeth_regression,
    walk_forward_split,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
log = logging.getLogger("Alpha01")

# ── configuration ─────────────────────────────────────────────────────────────
ALPHA_ID      = "01"
ALPHA_NAME    = "Reversal_VolDecay"
OUTPUT_DIR    = Path("./results")
REPORTS_DIR   = OUTPUT_DIR / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_START       = "2015-01-01"
DEFAULT_END         = "2024-12-31"
LAMBDA_GRID         = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
VOL_WINDOW          = 20          # rolling window for V̄
IC_LAGS             = [1, 2, 3, 5, 10, 22]
TOP_PCT             = 0.20        # long/short top-bottom 20%
TC_BPS              = 5.0         # round-trip transaction cost in basis points
IS_FRACTION         = 0.70        # 70% in-sample, 30% OOS


# ══════════════════════════════════════════════════════════════════════════════
class Alpha01:
    """
    Full implementation of Alpha 01 — Reversal with Volume Decay.

    Attributes
    ----------
    lam       : float   — best lambda chosen by IS IC maximisation
    signals   : pd.DataFrame  — daily (date × ticker) alpha signals
    returns   : pd.DataFrame  — daily (date × ticker) forward returns
    pnl       : pd.Series     — daily long-short portfolio returns (net of TC)
    ic_table  : pd.DataFrame  — IC statistics by lag
    metrics   : dict          — scalar performance metrics
    """

    def __init__(
        self,
        tickers:    List[str]   = None,
        start:      str         = DEFAULT_START,
        end:        str         = DEFAULT_END,
        lam:        float       = None,          # None → auto-tune
        vol_window: int         = VOL_WINDOW,
        ic_lags:    List[int]   = IC_LAGS,
        top_pct:    float       = TOP_PCT,
        tc_bps:     float       = TC_BPS,
        use_crypto: bool        = False,
    ):
        self.tickers    = tickers or (CRYPTO_UNIVERSE[:20] if use_crypto else SP500_TICKERS[:50])
        self.start      = start
        self.end        = end
        self.lam        = lam
        self.vol_window = vol_window
        self.ic_lags    = ic_lags
        self.top_pct    = top_pct
        self.tc_bps     = tc_bps
        self.use_crypto = use_crypto

        # output containers
        self.close:       Optional[pd.DataFrame] = None
        self.volume:      Optional[pd.DataFrame] = None
        self.returns:     Optional[pd.DataFrame] = None
        self.signals:     Optional[pd.DataFrame] = None
        self.naive_signals: Optional[pd.DataFrame] = None   # no volume adj.
        self.pnl:         Optional[pd.Series]    = None
        self.naive_pnl:   Optional[pd.Series]    = None
        self.ic_table:    Optional[pd.DataFrame] = None
        self.naive_ic_table: Optional[pd.DataFrame] = None
        self.metrics:     Dict                   = {}
        self.fm_result:   Dict                   = {}

        self._fetcher = DataFetcher()
        log.info("Alpha01 initialised | universe=%d tickers | %s→%s | λ=%s",
                 len(self.tickers), start, end, "auto" if lam is None else lam)

    # ── data loading ──────────────────────────────────────────────────────────

    def _load_data(self) -> None:
        """Download (or load from cache) close prices and volumes."""
        log.info("Loading price data …")
        if self.use_crypto:
            # crypto: Binance daily OHLCV
            ohlcv_dict = self._fetcher.get_crypto_universe_daily(self.tickers, self.start, self.end)
            close_frames  = {sym: df["Close"]  for sym, df in ohlcv_dict.items()}
            volume_frames = {sym: df["Volume"] for sym, df in ohlcv_dict.items()}
            self.close  = pd.DataFrame(close_frames).sort_index()
            self.volume = pd.DataFrame(volume_frames).sort_index()
        else:
            # equity: yfinance
            ohlcv_dict = self._fetcher.get_equity_ohlcv(self.tickers, self.start, self.end)
            close_frames  = {t: df["Close"]  for t, df in ohlcv_dict.items() if not df.empty}
            volume_frames = {t: df["Volume"] for t, df in ohlcv_dict.items() if not df.empty}
            self.close  = pd.DataFrame(close_frames).sort_index()
            self.volume = pd.DataFrame(volume_frames).sort_index()

        # align
        common_cols = self.close.columns.intersection(self.volume.columns)
        self.close  = self.close[common_cols].ffill().dropna(how="all", axis=1)
        self.volume = self.volume[common_cols].ffill().dropna(how="all", axis=1)

        # drop assets with <80% data coverage
        coverage = self.close.notna().mean()
        good     = coverage[coverage >= 0.80].index
        self.close  = self.close[good]
        self.volume = self.volume[good]

        log.info("Data loaded | %d assets | %d dates", self.close.shape[1], self.close.shape[0])

    # ── signal computation ────────────────────────────────────────────────────

    def _compute_signal(self, lam: float) -> pd.DataFrame:
        """
        Compute daily α₁ signal matrix for a given λ.

        α₁ = -rank(r_{t-1}) × exp(-λ × V_{t-1} / V̄₂₀)

        Returns DataFrame (date × ticker), values in [-1, +1].
        """
        # 1-day lagged returns
        ret_1d = np.log(self.close / self.close.shift(1)).shift(1)   # shift(1) → use yesterday's return

        # volume ratio: V_{t-1} / V̄₂₀   (today's data uses yesterday's volume)
        vol_lag    = self.volume.shift(1)
        vol_mean20 = vol_lag.rolling(self.vol_window, min_periods=5).mean()
        vol_ratio  = (vol_lag / vol_mean20.replace(0, np.nan)).clip(0, 10)

        # decay weight
        decay = np.exp(-lam * vol_ratio)

        # raw signal: negate return (reversal), scale by decay
        raw_signal = -ret_1d * decay

        # cross-sectional rank normalisation → [-1, +1]
        signal = cross_sectional_rank(raw_signal)

        return signal

    def _compute_naive_signal(self) -> pd.DataFrame:
        """Plain 1-day return reversal without volume adjustment."""
        ret_1d = np.log(self.close / self.close.shift(1)).shift(1)
        return cross_sectional_rank(-ret_1d)

    # ── lambda tuning ─────────────────────────────────────────────────────────

    def _tune_lambda(self, is_index: pd.DatetimeIndex) -> float:
        """
        Grid-search over LAMBDA_GRID using in-sample mean IC at lag-1.
        Returns the lambda that maximises IS IC.
        """
        log.info("Tuning λ over grid %s …", LAMBDA_GRID)
        returns_1d  = compute_returns(self.close, 1)
        best_lam    = LAMBDA_GRID[0]
        best_ic     = -np.inf

        for lam in LAMBDA_GRID:
            sig  = self._compute_signal(lam)
            sig_is  = sig.loc[sig.index.intersection(is_index)]
            ret_is  = returns_1d.loc[returns_1d.index.intersection(is_index)]
            ic_stats = information_coefficient_matrix(sig_is, ret_is, [1])
            ic_val   = ic_stats.loc[1, "mean_IC"] if 1 in ic_stats.index else -np.inf
            log.debug("  λ=%.2f → IS IC@1d = %.4f", lam, ic_val)
            if ic_val > best_ic:
                best_ic  = ic_val
                best_lam = lam

        log.info("Best λ = %.2f | IS IC@1d = %.4f", best_lam, best_ic)
        return best_lam

    # ── main pipeline ─────────────────────────────────────────────────────────

    def run(self) -> "Alpha01":
        """
        Full pipeline:  load data → tune λ → compute signals →
        backtest → compute IC tables → Fama-MacBeth → metrics.
        """
        self._load_data()

        self.returns = compute_returns(self.close, 1)

        # IS / OOS split
        is_idx, oos_idx = walk_forward_split(self.close.index, IS_FRACTION)

        # tune λ on IS data
        if self.lam is None:
            self.lam = self._tune_lambda(is_idx)

        # full signal computation
        self.signals       = self._compute_signal(self.lam)
        self.naive_signals = self._compute_naive_signal()

        # IC tables
        log.info("Computing IC tables …")
        self.ic_table = information_coefficient_matrix(
            self.signals, self.returns, self.ic_lags
        )

        # IS vs OOS IC breakdown
        self.ic_is = information_coefficient_matrix(
            self.signals.loc[self.signals.index.intersection(is_idx)],
            self.returns.loc[self.returns.index.intersection(is_idx)],
            self.ic_lags,
        )
        self.ic_oos = information_coefficient_matrix(
            self.signals.loc[self.signals.index.intersection(oos_idx)],
            self.returns.loc[self.returns.index.intersection(oos_idx)],
            self.ic_lags,
        )

        self.naive_ic_table = information_coefficient_matrix(
            self.naive_signals, self.returns, self.ic_lags
        )

        # Fama-MacBeth
        self.fm_result = fama_macbeth_regression(self.signals, self.returns, lag=1)
        log.info("Fama-MacBeth | γ=%.5f | t=%.2f", self.fm_result["gamma"], self.fm_result["t_stat"])

        # Portfolio PnL
        log.info("Computing portfolio PnL …")
        self.pnl = long_short_portfolio_returns(
            self.signals, self.returns, top_pct=self.top_pct, transaction_cost_bps=self.tc_bps
        )
        self.naive_pnl = long_short_portfolio_returns(
            self.naive_signals, self.returns, top_pct=self.top_pct, transaction_cost_bps=self.tc_bps
        )

        # summary metrics
        self._compute_metrics()
        return self

    # ── metrics ───────────────────────────────────────────────────────────────

    def _compute_metrics(self) -> None:
        pnl  = self.pnl.dropna()
        npnl = self.naive_pnl.dropna()

        ic1_is  = self.ic_is.loc[1, "mean_IC"]  if 1 in self.ic_is.index  else np.nan
        ic1_oos = self.ic_oos.loc[1, "mean_IC"] if 1 in self.ic_oos.index else np.nan

        self.metrics = {
            "alpha_id":         ALPHA_ID,
            "alpha_name":       ALPHA_NAME,
            "lambda":           self.lam,
            "n_assets":         self.close.shape[1],
            "n_dates":          self.close.shape[0],
            "IS_start":         str(self.close.index[0].date()),
            "OOS_start":        str(self.close.index[int(len(self.close) * IS_FRACTION)].date()),
            "end":              str(self.close.index[-1].date()),
            # IC
            "IC_mean_IS_lag1":  float(ic1_is),
            "IC_mean_OOS_lag1": float(ic1_oos),
            "ICIR_IS":          float(self.ic_is.loc[1, "ICIR"])  if 1 in self.ic_is.index  else np.nan,
            "ICIR_OOS":         float(self.ic_oos.loc[1, "ICIR"]) if 1 in self.ic_oos.index else np.nan,
            # Fama-MacBeth
            "FM_gamma":         float(self.fm_result["gamma"]),
            "FM_t_stat":        float(self.fm_result["t_stat"]),
            # portfolio stats (vol-adjusted)
            "Sharpe":           compute_sharpe(pnl),
            "Naive_Sharpe":     compute_sharpe(npnl),
            "MaxDrawdown":      compute_max_drawdown(pnl),
            "Annualised_Return":float(pnl.mean() * 252),
            "Turnover":         compute_turnover(self.signals),
            # IC comparison
            "IC_lift_vs_naive": float(ic1_oos - (self.naive_ic_table.loc[1, "mean_IC"] if 1 in self.naive_ic_table.index else 0)),
        }

        log.info("─── Alpha 01 Metrics ───────────────────────────────────")
        for k, v in self.metrics.items():
            log.info("  %-30s = %s", k, f"{v:.5f}" if isinstance(v, float) else v)

    # ── Kupiec test ───────────────────────────────────────────────────────────

    def kupiec_test(self, confidence: float = 0.95) -> Dict[str, float]:
        """
        Kupiec Proportion of Failures (POF) test on the long-short portfolio.
        Tests whether the frequency of losses exceeding VaR is consistent with
        the stated confidence level.
        """
        from scipy.stats import chi2

        pnl  = self.pnl.dropna()
        var  = pnl.quantile(1 - confidence)
        T    = len(pnl)
        n    = (pnl < var).sum()
        p    = 1 - confidence
        p_hat = n / T

        if p_hat == 0:
            return {"kupiec_stat": 0.0, "p_value": 1.0, "VaR": float(var)}

        lr = -2 * (n * np.log(p / p_hat) + (T - n) * np.log((1 - p) / (1 - p_hat)))
        p_value = 1 - chi2.cdf(lr, df=1)

        return {
            "kupiec_stat":  float(lr),
            "p_value":      float(p_value),
            "VaR_95":       float(var),
            "n_violations": int(n),
            "T":            int(T),
        }

    # ── visualisation ─────────────────────────────────────────────────────────

    def plot(self, save: bool = True) -> None:
        """
        4-panel figure:
        1. IC decay curve (vol-adj vs naive)
        2. Cumulative PnL (vol-adj vs naive vs zero)
        3. IC IS vs OOS by lag
        4. Monthly IC heatmap
        """
        fig = plt.figure(figsize=(18, 14))
        gs  = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.30)

        # ── Panel 1: IC Decay ─────────────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        lags_plot = [l for l in self.ic_lags if l in self.ic_table.index]
        ic_vals     = [self.ic_table.loc[l,       "mean_IC"] for l in lags_plot]
        ic_naive    = [self.naive_ic_table.loc[l,  "mean_IC"] for l in lags_plot if l in self.naive_ic_table.index]

        ax1.plot(lags_plot, ic_vals,  marker="o", linewidth=2, label=f"Vol-Adj (λ={self.lam:.2f})", color="#1f77b4")
        if ic_naive:
            ax1.plot(lags_plot[:len(ic_naive)], ic_naive, marker="s", linewidth=2, linestyle="--",
                     label="Naive Reversal", color="#ff7f0e", alpha=0.8)
        ax1.axhline(0, color="black", linewidth=0.8, linestyle=":")
        ax1.fill_between(lags_plot, ic_vals, 0, alpha=0.12, color="#1f77b4")
        ax1.set_xlabel("Lag (days)")
        ax1.set_ylabel("Mean IC")
        ax1.set_title("Alpha 01 — IC Decay Curve\n(Vol-Adjusted vs Naive Reversal)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # ── Panel 2: Cumulative PnL ───────────────────────────────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        cum_adj   = self.pnl.dropna().cumsum()
        cum_naive = self.naive_pnl.dropna().cumsum()
        ax2.plot(cum_adj.index,   cum_adj.values,   linewidth=1.8, label="Vol-Adj Signal", color="#1f77b4")
        ax2.plot(cum_naive.index, cum_naive.values, linewidth=1.8, linestyle="--",
                 label="Naive Reversal", color="#ff7f0e", alpha=0.8)
        ax2.axhline(0, color="black", linewidth=0.6)
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Cumulative Return")
        ax2.set_title("Alpha 01 — Cumulative Long-Short PnL\n(Net of 5 bps TC)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # ── Panel 3: IS vs OOS IC by Lag ─────────────────────────────────────
        ax3 = fig.add_subplot(gs[1, 0])
        lags_common = sorted(set(self.ic_is.index) & set(self.ic_oos.index))
        x = np.arange(len(lags_common))
        w = 0.35
        ic_is_vals  = [self.ic_is.loc[l,  "mean_IC"] for l in lags_common]
        ic_oos_vals = [self.ic_oos.loc[l, "mean_IC"] for l in lags_common]
        ax3.bar(x - w/2, ic_is_vals,  w, label="In-Sample",     color="#2ca02c", alpha=0.8)
        ax3.bar(x + w/2, ic_oos_vals, w, label="Out-of-Sample", color="#d62728", alpha=0.8)
        ax3.set_xticks(x)
        ax3.set_xticklabels([f"Lag {l}" for l in lags_common])
        ax3.axhline(0, color="black", linewidth=0.6)
        ax3.set_ylabel("Mean IC")
        ax3.set_title("Alpha 01 — IS vs OOS IC by Lag")
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis="y")

        # ── Panel 4: Monthly IC Heatmap ───────────────────────────────────────
        ax4 = fig.add_subplot(gs[1, 1])
        # compute monthly IC at lag=1
        returns_1d = compute_returns(self.close, 1)
        monthly_ic = []
        for date, row in self.signals.iterrows():
            if date not in returns_1d.index:
                continue
            fwd_date = date + pd.Timedelta(days=1)
            matches  = returns_1d.index[returns_1d.index >= fwd_date]
            if len(matches) == 0:
                continue
            fwd_row  = returns_1d.loc[matches[0]]
            common   = row.dropna().index.intersection(fwd_row.dropna().index)
            if len(common) < 5:
                continue
            ic = information_coefficient(row[common], fwd_row[common])
            monthly_ic.append({"date": date, "IC": ic})

        if monthly_ic:
            ic_series = pd.DataFrame(monthly_ic).set_index("date")["IC"]
            ic_monthly_mean = ic_series.resample("ME").mean()
            years   = ic_monthly_mean.index.year.unique()
            months  = list(range(1, 13))
            heat_data = np.full((len(years), 12), np.nan)
            year_list = sorted(years)
            for i, yr in enumerate(year_list):
                for j, mo in enumerate(months):
                    mask = (ic_monthly_mean.index.year == yr) & (ic_monthly_mean.index.month == mo)
                    if mask.any():
                        heat_data[i, j] = ic_monthly_mean[mask].values[0]

            vmax = np.nanpercentile(np.abs(heat_data), 90)
            im = ax4.imshow(heat_data, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
            plt.colorbar(im, ax=ax4, label="Mean IC")
            ax4.set_xticks(range(12))
            ax4.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
                                 fontsize=7)
            ax4.set_yticks(range(len(year_list)))
            ax4.set_yticklabels(year_list, fontsize=7)
            ax4.set_title("Alpha 01 — Monthly IC Heatmap (Lag=1)")

        plt.suptitle(
            f"ALPHA 01 — Reversal with Volume Decay\n"
            f"Sharpe={self.metrics.get('Sharpe', np.nan):.2f}  "
            f"IC_IS(1d)={self.metrics.get('IC_mean_IS_lag1', np.nan):.4f}  "
            f"IC_OOS(1d)={self.metrics.get('IC_mean_OOS_lag1', np.nan):.4f}  "
            f"FM t-stat={self.metrics.get('FM_t_stat', np.nan):.2f}",
            fontsize=13, fontweight="bold",
        )

        if save:
            out_path = REPORTS_DIR / f"alpha_{ALPHA_ID}_chart.png"
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            log.info("Chart saved → %s", out_path)
        plt.close(fig)

    # ── markdown report ───────────────────────────────────────────────────────

    def generate_report(self) -> str:
        """
        Auto-generates a Markdown performance report and writes it to disk.
        Returns the Markdown string.
        """
        kupiec = self.kupiec_test()

        ic_table_str = self.ic_table.reset_index().to_markdown(index=False, floatfmt=".5f")
        naive_str    = self.naive_ic_table.reset_index().to_markdown(index=False, floatfmt=".5f")

        ic_is_str   = self.ic_is.reset_index().to_markdown(index=False, floatfmt=".5f")
        ic_oos_str  = self.ic_oos.reset_index().to_markdown(index=False, floatfmt=".5f")

        report = f"""# Alpha {ALPHA_ID}: {ALPHA_NAME.replace('_', ' ')}

## Hypothesis
Short-term price reversals are weaker when accompanied by abnormally high volume
(suggesting informed flow), and stronger in low-volume environments (noise-driven).
Volume-decaying the reversal signal improves IC by filtering out momentum driven by
information.

## Expression (Python)
```python
ret_1d     = -log(close / close.shift(1)).shift(1)          # lag-1 return, sign-flipped
vol_ratio  = volume.shift(1) / volume.shift(1).rolling(20).mean()
decay      = exp(-{self.lam:.2f} * vol_ratio)
alpha_01   = cross_sectional_rank(ret_1d * decay)           # [-1, +1]
```

## Parameters
| Parameter       | Value             |
|-----------------|-------------------|
| Lambda (λ)      | {self.lam:.2f}    |
| Volume window   | {self.vol_window} days |
| Rebalance       | Daily             |
| Long/Short pct  | {self.top_pct*100:.0f}% |
| TC assumption   | {self.tc_bps:.0f} bps round-trip |

## Performance Summary

| Metric               | Vol-Adj Signal | Naive Reversal |
|----------------------|----------------|----------------|
| Sharpe Ratio         | {self.metrics.get('Sharpe', np.nan):.3f}  | {self.metrics.get('Naive_Sharpe', np.nan):.3f}  |
| Annualised Return    | {self.metrics.get('Annualised_Return', np.nan)*100:.2f}%  | — |
| Max Drawdown         | {self.metrics.get('MaxDrawdown', np.nan)*100:.2f}%  | — |
| IC (IS) @ Lag 1      | {self.metrics.get('IC_mean_IS_lag1', np.nan):.5f}  | {self.naive_ic_table.loc[1, 'mean_IC'] if 1 in self.naive_ic_table.index else np.nan:.5f} |
| IC (OOS) @ Lag 1     | {self.metrics.get('IC_mean_OOS_lag1', np.nan):.5f}  | — |
| ICIR (IS)            | {self.metrics.get('ICIR_IS', np.nan):.3f}  | — |
| ICIR (OOS)           | {self.metrics.get('ICIR_OOS', np.nan):.3f}  | — |
| Fama-MacBeth γ       | {self.metrics.get('FM_gamma', np.nan):.6f}  | — |
| Fama-MacBeth t-stat  | {self.metrics.get('FM_t_stat', np.nan):.3f}  | — |
| IC Lift vs Naive     | {self.metrics.get('IC_lift_vs_naive', np.nan):+.5f}  | — |
| Daily Turnover       | {self.metrics.get('Turnover', np.nan)*100:.1f}%  | — |

## IC Decay Table (Full Sample)
{ic_table_str}

### Naive Reversal IC Decay (Comparison)
{naive_str}

## In-Sample IC by Lag
{ic_is_str}

## Out-of-Sample IC by Lag
{ic_oos_str}

## Fama-MacBeth Regression (Lag = 1)
| Statistic   | Value   |
|-------------|---------|
| γ (slope)   | {self.fm_result['gamma']:.6f} |
| t-statistic | {self.fm_result['t_stat']:.3f} |
| n_periods   | {self.fm_result['n_periods']} |

> **Interpretation**: t-stat > 2.0 rejects the null of zero cross-sectional predictability.

## Kupiec Tail-Risk Test (95% VaR)
| Statistic       | Value   |
|-----------------|---------|
| LR Statistic    | {kupiec['kupiec_stat']:.3f} |
| p-value         | {kupiec['p_value']:.4f} |
| VaR (95%)       | {kupiec['VaR_95']*100:.4f}% |
| # Violations    | {kupiec['n_violations']} / {kupiec['T']} |

> **Interpretation**: p-value > 0.05 means the model's VaR coverage is statistically adequate.

## Regime-Conditional IC
*See Alpha 09 (HMM Factor Rotation) for regime-conditional breakdown.*

## Correlation to Other Alphas
*Populated by alpha_reporter.py during combined portfolio construction.*

## Transaction Cost Break-Even
Minimum spread required for positive net-of-cost return:
`{self.tc_bps:.0f} bps` (current assumption).
Break-even spread = Gross Return / (Turnover × Days) ≈
`{(self.pnl.dropna().mean() / (self.metrics.get('Turnover', 0.1) + 1e-9))*10_000:.1f} bps per trade`.

## Academic Reference
- Jegadeesh (1990) *Evidence of Predictable Behavior of Security Returns* — JF
- Avramov, Chordia & Goyal (2006) *Liquidity and Autocorrelations in Individual Stock Returns* — JF
- Lehmann (1990) *Fads, Martingales, and Market Efficiency* — QJE
"""
        report_path = REPORTS_DIR / f"alpha_{ALPHA_ID}_report.md"
        report_path.write_text(report)
        log.info("Report saved → %s", report_path)
        return report


# ══════════════════════════════════════════════════════════════════════════════
#  Standalone execution / diagnostic
# ══════════════════════════════════════════════════════════════════════════════

def run_alpha01(
    tickers:    List[str] = None,
    start:      str       = DEFAULT_START,
    end:        str       = DEFAULT_END,
    use_crypto: bool      = False,
) -> Alpha01:
    """Convenience entry-point — run the full pipeline and return the object."""
    alpha = Alpha01(
        tickers    = tickers,
        start      = start,
        end        = end,
        use_crypto = use_crypto,
    )
    alpha.run()
    alpha.plot()
    alpha.generate_report()

    # save metrics row to results CSV
    metrics_csv = OUTPUT_DIR / "alpha_performance_summary.csv"
    row = pd.DataFrame([alpha.metrics])
    if metrics_csv.exists():
        existing = pd.read_csv(metrics_csv, index_col=0)
        existing = existing[existing["alpha_id"] != ALPHA_ID]
        row = pd.concat([existing, row], ignore_index=True)
    row.to_csv(metrics_csv)
    log.info("Metrics appended → %s", metrics_csv)

    return alpha


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Alpha 01 — Reversal with Volume Decay")
    parser.add_argument("--start",      default=DEFAULT_START, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",        default=DEFAULT_END,   help="End date YYYY-MM-DD")
    parser.add_argument("--crypto",     action="store_true",   help="Use crypto universe instead of equity")
    parser.add_argument("--lam",        type=float, default=None, help="Override lambda (skip tuning)")
    args = parser.parse_args()

    alpha = Alpha01(
        start      = args.start,
        end        = args.end,
        lam        = args.lam,
        use_crypto = args.crypto,
    )
    alpha.run()
    alpha.plot()
    alpha.generate_report()

    print("\n" + "=" * 60)
    print("ALPHA 01 COMPLETE")
    print("=" * 60)
    for k, v in alpha.metrics.items():
        val = f"{v:.5f}" if isinstance(v, float) else str(v)
        print(f"  {k:<35} {val}")
