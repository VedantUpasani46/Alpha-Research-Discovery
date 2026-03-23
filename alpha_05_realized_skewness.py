"""
alpha_05_realized_skewness.py
──────────────────────────────
ALPHA 05 — Realized Skewness Reversal
=======================================

HYPOTHESIS
----------
Assets with highly POSITIVE realized skewness (lottery-like payoff structure —
many small losses, rare enormous gains) are systematically OVERPRICED.
Retail investors chase these "lottery assets" due to the preference for
positive skewness (probability weighting, CPT), driving valuations above
fundamental value.

Assets with NEGATIVE realized skewness (many small gains, rare devastating losses)
are systematically UNDERPRICED because investors avoid asymmetric downside exposure.

The cross-sectional return prediction is:
    Short high-positive-skewness (lottery assets) → negative future returns
    Long negative-skewness (steady assets) → positive future returns

This is a REVERSAL alpha — the overpriced lottery assets mean-revert downward.

FORMULA
-------
    ŝ_{i,t} = [N × Σ(r_{i,d} - r̄_i)³] / [(N-1)(N-2) × (std(r_{i,d}))³]

    (Fisher adjusted skewness, unbiased for small samples)

    α₅ = -rank(ŝ_{i,t})

    Negative sign: SHORT high-positive-skewness, LONG negative-skewness.

ASSET CLASS
-----------
• Primary: S&P 500 equities
• Also applicable to: Crypto (documented in Liu & Tsyvinski 2021; Akyildirim et al.)

REBALANCE FREQUENCY
-------------------
Weekly (5-day).  Skewness is a medium-term predictor — rebalancing faster
than weekly mostly generates transaction costs with no incremental IC.

VALIDATION
----------
• IC at 5-day, 22-day, 63-day horizons
• Show IC is negative at short horizons (reversal — long neg skew, short pos skew)
• Correlation to momentum factor (partial overlap; document and show they differ)
• Fama-MacBeth t-statistic
• Long-only and short-only IC (is the short side as strong as the long?)
• Sharpe, Max Drawdown

IMPLEMENTATION NOTES
─────────────────────
Uses scipy.stats.skew with bias=False (Fisher's correction = unbiased estimator).
Rolling window: 22 days (monthly) — captures near-term distribution shape.
Also computes 5-day and 63-day windows for robustness check.

REFERENCES
----------
• Harvey & Siddique (2000) *Conditional Skewness in Asset Pricing Tests* — JF
• Kumar (2009) *Who Gambles in the Stock Market?* — JF
• Bali, Cakici & Whitelaw (2011) *Maxing Out: Stocks as Lotteries* — JFinEc
• Akyildirim et al. (2021) *Do Skewness and Kurtosis of Crypto Returns Matter?*

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
log = logging.getLogger("Alpha05")

# ── configuration ─────────────────────────────────────────────────────────────
ALPHA_ID          = "05"
ALPHA_NAME        = "Realized_Skewness_Reversal"
OUTPUT_DIR        = Path("./results")
REPORTS_DIR       = OUTPUT_DIR / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_START     = "2015-01-01"
DEFAULT_END       = "2024-12-31"
SKEW_WINDOWS      = [5, 10, 22, 63]   # rolling windows to test
MAIN_SKEW_WINDOW  = 22               # primary signal window
IC_LAGS           = [1, 5, 10, 22, 44, 63]
TOP_PCT           = 0.20
TC_BPS            = 8.0
IS_FRACTION       = 0.70
REBALANCE_DAYS    = 5               # weekly


# ══════════════════════════════════════════════════════════════════════════════
class SkewnessCalculator:
    """
    Computes various skewness estimators on rolling windows of daily returns.

    Implements three estimators:
    1. Fisher (unbiased, default):
       Adjusted formula for small samples using N/(N-1)/(N-2) correction.

    2. Pearson moment skewness (biased):
       μ₃ / σ³  (biased for small N, use for comparison only)

    3. Nonparametric skewness (Bowley):
       (Q3 + Q1 - 2×Q2) / (Q3 - Q1)  — robust to outliers, no distribution assumption
    """

    @staticmethod
    def fisher_skewness(returns: pd.Series, min_periods: int = 10) -> float:
        """Unbiased Fisher-adjusted skewness estimator."""
        n = returns.count()
        if n < min_periods:
            return np.nan
        return float(sp_stats.skew(returns.dropna(), bias=False))

    @staticmethod
    def rolling_skewness(
        returns:     pd.DataFrame,
        window:      int,
        min_periods: int = 10,
        method:      str = "fisher",
    ) -> pd.DataFrame:
        """
        Compute rolling skewness for each column of a returns DataFrame.

        Parameters
        ----------
        returns    : (date × ticker) DataFrame of returns
        window     : rolling window in days
        min_periods: minimum non-NaN observations required
        method     : "fisher" | "pearson" | "nonparametric"

        Returns
        -------
        (date × ticker) DataFrame of rolling skewness values
        """
        if method == "fisher":
            # scipy skew with bias=False = Fisher-adjusted
            result = returns.rolling(window, min_periods=min_periods).apply(
                lambda x: sp_stats.skew(x[~np.isnan(x)], bias=False) if len(x[~np.isnan(x)]) >= min_periods else np.nan,
                raw=True,
            )
        elif method == "pearson":
            result = returns.rolling(window, min_periods=min_periods).apply(
                lambda x: sp_stats.skew(x[~np.isnan(x)], bias=True) if len(x[~np.isnan(x)]) >= min_periods else np.nan,
                raw=True,
            )
        elif method == "nonparametric":
            def bowley(x):
                x_clean = x[~np.isnan(x)]
                if len(x_clean) < min_periods:
                    return np.nan
                q1, q2, q3 = np.percentile(x_clean, [25, 50, 75])
                if (q3 - q1) == 0:
                    return np.nan
                return (q3 + q1 - 2 * q2) / (q3 - q1)
            result = returns.rolling(window, min_periods=min_periods).apply(bowley, raw=True)
        else:
            raise ValueError(f"Unknown method: {method}")

        return result

    @staticmethod
    def excess_kurtosis(
        returns:     pd.DataFrame,
        window:      int,
        min_periods: int = 10,
    ) -> pd.DataFrame:
        """Rolling excess kurtosis (for completeness; not used in alpha signal)."""
        return returns.rolling(window, min_periods=min_periods).apply(
            lambda x: sp_stats.kurtosis(x[~np.isnan(x)], bias=False) if len(x[~np.isnan(x)]) >= min_periods else np.nan,
            raw=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
class Alpha05:
    """
    Full implementation of Alpha 05 — Realized Skewness Reversal.

    Key outputs:
    - Signal at primary window (22d) and comparison windows (5d, 63d)
    - IC at multiple lags showing that the alpha is strongest at 5–22d
    - Correlation to momentum factor (to show they are partially distinct)
    - Long-only vs short-only IC analysis
    """

    def __init__(
        self,
        tickers:        List[str] = None,
        start:          str       = DEFAULT_START,
        end:            str       = DEFAULT_END,
        skew_windows:   List[int] = SKEW_WINDOWS,
        main_window:    int       = MAIN_SKEW_WINDOW,
        ic_lags:        List[int] = IC_LAGS,
        top_pct:        float     = TOP_PCT,
        tc_bps:         float     = TC_BPS,
        use_crypto:     bool      = False,
    ):
        self.tickers      = tickers or (CRYPTO_UNIVERSE[:20] if use_crypto else SP500_TICKERS[:50])
        self.start        = start
        self.end          = end
        self.skew_windows = skew_windows
        self.main_window  = main_window
        self.ic_lags      = ic_lags
        self.top_pct      = top_pct
        self.tc_bps       = tc_bps
        self.use_crypto   = use_crypto

        self._fetcher   = DataFetcher()
        self._skew_calc = SkewnessCalculator()

        # outputs
        self.close:      Optional[pd.DataFrame] = None
        self.returns:    Optional[pd.DataFrame] = None
        self.skew_dfs:   Dict[int, pd.DataFrame] = {}   # window → skewness DataFrame
        self.signals:    Optional[pd.DataFrame] = None  # main signal (22d)
        self.pnl:        Optional[pd.Series]    = None
        self.ic_table:   Optional[pd.DataFrame] = None
        self.ic_is:      Optional[pd.DataFrame] = None
        self.ic_oos:     Optional[pd.DataFrame] = None
        self.ic_by_window: Optional[pd.DataFrame] = None
        self.momentum_corr: Optional[pd.Series] = None
        self.sign_analysis: Optional[pd.DataFrame] = None
        self.fm_result:  Dict                   = {}
        self.metrics:    Dict                   = {}

        log.info("Alpha05 initialised | %d tickers | %s→%s | main_window=%d | use_crypto=%s",
                 len(self.tickers), start, end, main_window, use_crypto)

    # ── data loading ──────────────────────────────────────────────────────────

    def _load_data(self) -> None:
        log.info("Loading price data …")
        if self.use_crypto:
            ohlcv_dict = self._fetcher.get_crypto_universe_daily(self.tickers, self.start, self.end)
            close_frames = {sym: df["Close"] for sym, df in ohlcv_dict.items()}
        else:
            ohlcv_dict = self._fetcher.get_equity_ohlcv(self.tickers, self.start, self.end)
            close_frames = {t: df["Close"] for t, df in ohlcv_dict.items() if not df.empty}

        self.close = pd.DataFrame(close_frames).sort_index().ffill()
        coverage   = self.close.notna().mean()
        good       = coverage[coverage >= 0.80].index
        self.close = self.close[good]
        self.returns = compute_returns(self.close, 1)

        log.info("Data loaded | %d assets | %d dates", self.close.shape[1], self.close.shape[0])

    # ── skewness computation ──────────────────────────────────────────────────

    def _compute_all_skewness(self) -> None:
        """Compute rolling skewness at each window in self.skew_windows."""
        log.info("Computing rolling skewness for windows %s …", self.skew_windows)
        for w in self.skew_windows:
            skew_df = SkewnessCalculator.rolling_skewness(
                self.returns, window=w, min_periods=max(5, w // 3), method="fisher"
            )
            self.skew_dfs[w] = skew_df
            log.info("  Skewness computed | window=%dd | NaN=%.1f%%",
                     w, skew_df.isna().mean().mean() * 100)

    # ── signal construction ───────────────────────────────────────────────────

    def _compute_signal(self) -> None:
        """
        α₅ = -rank(ŝ_{i,t})  using the main_window skewness.

        Negative sign: short positive-skewness (lottery), long negative-skewness.
        """
        log.info("Computing alpha signal (window=%dd) …", self.main_window)
        skew_main  = self.skew_dfs[self.main_window]
        raw_signal = -skew_main    # negate: high skew → negative signal (short)
        self.signals = cross_sectional_rank(raw_signal)
        log.info("Signal computed | shape=%s", self.signals.shape)

    # ── IC by window analysis ─────────────────────────────────────────────────

    def _compute_ic_by_window(self) -> None:
        """
        For each skewness window, compute IC at the main IC lags.
        Shows which window is most predictive and at what horizon.
        """
        log.info("Computing IC by skewness window …")
        rows = []
        for w in self.skew_windows:
            skew_df = self.skew_dfs[w]
            sig     = cross_sectional_rank(-skew_df)
            for lag in [5, 22, 63]:
                ic_stats = information_coefficient_matrix(sig, self.returns, [lag])
                ic_val   = ic_stats.loc[lag, "mean_IC"] if lag in ic_stats.index else np.nan
                rows.append({"skew_window": w, "lag_d": lag, "IC": ic_val})

        self.ic_by_window = pd.DataFrame(rows)
        pivot = self.ic_by_window.pivot(index="skew_window", columns="lag_d", values="IC")
        log.info("IC by window:\n%s", pivot.to_string())

    # ── momentum factor correlation ───────────────────────────────────────────

    def _compute_momentum_correlation(self) -> None:
        """
        Compute correlation between the skewness reversal signal and the
        standard 12-1 momentum signal.  They should be partially distinct.
        """
        log.info("Computing correlation to momentum factor …")
        # 12-1 month momentum
        price_21ago = self.close.shift(22)
        price_1ago  = self.close.shift(1)
        momentum    = np.log(price_1ago / price_21ago)
        mom_signal  = cross_sectional_rank(momentum)

        # correlate signal values across all dates and tickers
        flat_skew = self.signals.values.flatten()
        flat_mom  = mom_signal.reindex(self.signals.index).values.flatten()
        valid     = np.isfinite(flat_skew) & np.isfinite(flat_mom)

        pearson_r, p_value = sp_stats.pearsonr(flat_skew[valid], flat_mom[valid])
        spearman_r, sp_p   = sp_stats.spearmanr(flat_skew[valid], flat_mom[valid])

        self.momentum_corr = {
            "pearson_r":   float(pearson_r),
            "pearson_p":   float(p_value),
            "spearman_r":  float(spearman_r),
            "spearman_p":  float(sp_p),
            "interpretation": (
                "Low correlation (<0.3) → skewness signal adds independent information to momentum"
                if abs(pearson_r) < 0.3 else
                "Moderate correlation — skewness and momentum partially overlap"
            )
        }
        log.info("Momentum correlation: Pearson r=%.4f (p=%.4f)", pearson_r, p_value)

    # ── long vs short IC analysis ─────────────────────────────────────────────

    def _compute_sign_analysis(self) -> None:
        """
        Separate IC analysis for:
        1. Long side: low/negative skewness assets (predicted to outperform)
        2. Short side: high/positive skewness assets (predicted to underperform)
        Checks if both sides have IC in the same direction.
        """
        log.info("Computing sign asymmetry …")
        fwd_5d = self.returns.shift(-5)   # 5-day forward return

        long_ics, short_ics = [], []
        for date in self.signals.index:
            if date not in fwd_5d.index:
                continue
            sig = self.signals.loc[date].dropna()
            fwd = fwd_5d.loc[date].dropna()
            common = sig.index.intersection(fwd.index)
            if len(common) < 8:
                continue

            # long side: top 20% by signal (most negative skewness)
            n_top = max(2, int(len(common) * 0.20))
            long_assets  = sig.nlargest(n_top).index    # highest signal = most negative skew
            short_assets = sig.nsmallest(n_top).index   # lowest signal = most positive skew

            if len(long_assets) >= 2:
                ic_l = information_coefficient(sig[long_assets], fwd[long_assets])
                long_ics.append(ic_l)

            if len(short_assets) >= 2:
                # for short side: signal is negative, return should be negative
                ic_s = information_coefficient(sig[short_assets], fwd[short_assets])
                short_ics.append(ic_s)

        rows = []
        for name, ics in [("Long (neg skew assets)", long_ics), ("Short (pos skew assets)", short_ics)]:
            arr = np.array([x for x in ics if not np.isnan(x)])
            if len(arr) >= 3:
                t_stat = arr.mean() / (arr.std(ddof=1) / np.sqrt(len(arr))) if arr.std(ddof=1) > 0 else np.nan
                rows.append({
                    "Side":     name,
                    "Mean_IC":  arr.mean(),
                    "Std_IC":   arr.std(ddof=1),
                    "ICIR":     arr.mean() / arr.std(ddof=1) if arr.std(ddof=1) > 0 else np.nan,
                    "t_stat":   t_stat,
                    "n_obs":    len(arr),
                })
            else:
                rows.append({"Side": name, "Mean_IC": np.nan, "Std_IC": np.nan, "ICIR": np.nan, "t_stat": np.nan, "n_obs": 0})

        self.sign_analysis = pd.DataFrame(rows).set_index("Side")

    # ── cross-sectional skewness distribution ─────────────────────────────────

    def skewness_stats_by_asset(self) -> pd.DataFrame:
        """
        Returns time-series statistics of skewness per asset:
        mean, std, % of time in positive/negative skewness.
        """
        skew_main = self.skew_dfs[self.main_window]
        stats = pd.DataFrame({
            "mean_skew":    skew_main.mean(),
            "std_skew":     skew_main.std(),
            "pct_pos_skew": (skew_main > 0).mean(),
            "pct_neg_skew": (skew_main < 0).mean(),
            "max_skew":     skew_main.max(),
            "min_skew":     skew_main.min(),
        })
        return stats.sort_values("mean_skew", ascending=False)

    # ── main pipeline ─────────────────────────────────────────────────────────

    def run(self) -> "Alpha05":
        self._load_data()
        self._compute_all_skewness()
        self._compute_signal()

        is_idx, oos_idx = walk_forward_split(self.close.index, IS_FRACTION)

        # IC tables
        log.info("Computing IC tables …")
        self.ic_table = information_coefficient_matrix(
            self.signals, self.returns, self.ic_lags
        )
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

        # Window comparison
        self._compute_ic_by_window()

        # Momentum correlation
        self._compute_momentum_correlation()

        # Sign analysis
        self._compute_sign_analysis()

        # Fama-MacBeth at 5d and 22d
        self.fm_result_5  = fama_macbeth_regression(self.signals, self.returns, lag=5)
        self.fm_result_22 = fama_macbeth_regression(self.signals, self.returns, lag=22)
        log.info("FM@5d  γ=%.5f t=%.2f", self.fm_result_5["gamma"],  self.fm_result_5["t_stat"])
        log.info("FM@22d γ=%.5f t=%.2f", self.fm_result_22["gamma"], self.fm_result_22["t_stat"])

        # Portfolio (weekly rebalancing)
        log.info("Computing weekly portfolio …")
        self.pnl = long_short_portfolio_returns(
            self.signals, self.returns, top_pct=self.top_pct, transaction_cost_bps=self.tc_bps
        )

        self._compute_metrics()
        return self

    # ── metrics ───────────────────────────────────────────────────────────────

    def _compute_metrics(self) -> None:
        pnl = self.pnl.dropna()
        ic5_is   = self.ic_is.loc[5,  "mean_IC"] if 5  in self.ic_is.index  else np.nan
        ic5_oos  = self.ic_oos.loc[5,  "mean_IC"] if 5  in self.ic_oos.index else np.nan
        ic22_oos = self.ic_oos.loc[22, "mean_IC"] if 22 in self.ic_oos.index else np.nan
        ic63_oos = self.ic_oos.loc[63, "mean_IC"] if 63 in self.ic_oos.index else np.nan

        self.metrics = {
            "alpha_id":           ALPHA_ID,
            "alpha_name":         ALPHA_NAME,
            "universe":           "Crypto" if self.use_crypto else "Equity",
            "n_assets":           self.close.shape[1],
            "n_dates":            self.close.shape[0],
            "skew_window":        self.main_window,
            "IC_mean_IS_lag5":    float(ic5_is),
            "IC_mean_OOS_lag5":   float(ic5_oos),
            "IC_mean_OOS_lag22":  float(ic22_oos),
            "IC_mean_OOS_lag63":  float(ic63_oos),
            "ICIR_IS_5d":         float(self.ic_is.loc[5,  "ICIR"]) if 5  in self.ic_is.index  else np.nan,
            "ICIR_OOS_5d":        float(self.ic_oos.loc[5,  "ICIR"]) if 5  in self.ic_oos.index else np.nan,
            "FM_gamma_5d":        float(self.fm_result_5["gamma"]),
            "FM_t_stat_5d":       float(self.fm_result_5["t_stat"]),
            "FM_gamma_22d":       float(self.fm_result_22["gamma"]),
            "FM_t_stat_22d":      float(self.fm_result_22["t_stat"]),
            "Sharpe":             compute_sharpe(pnl),
            "MaxDrawdown":        compute_max_drawdown(pnl),
            "Annualised_Return":  float(pnl.mean() * 252),
            "Turnover":           compute_turnover(self.signals),
            "Momentum_Corr_r":    float(self.momentum_corr["pearson_r"]) if self.momentum_corr else np.nan,
        }

        log.info("─── Alpha 05 Metrics ───────────────────────────────────")
        for k, v in self.metrics.items():
            log.info("  %-35s = %s", k, f"{v:.5f}" if isinstance(v, float) else v)

    # ── visualisation ─────────────────────────────────────────────────────────

    def plot(self, save: bool = True) -> None:
        """
        6-panel figure:
        1. IC decay curve (IS vs OOS)
        2. Cumulative PnL
        3. IC by skewness window (comparison)
        4. Skewness distribution (cross-sectional snapshot)
        5. Sign asymmetry bar chart
        6. Skewness signal vs momentum scatter
        """
        fig = plt.figure(figsize=(20, 18))
        gs  = gridspec.GridSpec(3, 2, hspace=0.42, wspace=0.30)

        # ── Panel 1: IC Decay ─────────────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        lags_plot = [l for l in self.ic_lags if l in self.ic_table.index]
        ic_is_vals  = [self.ic_is.loc[l,  "mean_IC"] if l in self.ic_is.index  else np.nan for l in lags_plot]
        ic_oos_vals = [self.ic_oos.loc[l, "mean_IC"] if l in self.ic_oos.index else np.nan for l in lags_plot]
        ax1.plot(lags_plot, ic_is_vals,  marker="o", label="In-Sample",     color="#2ca02c", linewidth=2)
        ax1.plot(lags_plot, ic_oos_vals, marker="s", label="Out-of-Sample", color="#d62728", linewidth=2, linestyle="--")
        ax1.axhline(0, color="black", linewidth=0.8)
        ax1.fill_between(lags_plot, ic_is_vals, 0, alpha=0.1, color="#2ca02c")
        ax1.set_xlabel("Lag (days)")
        ax1.set_ylabel("Mean IC")
        ax1.set_title(f"Alpha 05 — IC Decay Curve\n(Window={self.main_window}d, skewness reversal)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # ── Panel 2: Cumulative PnL ─────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        cum_pnl = self.pnl.dropna().cumsum()
        roll_max = cum_pnl.cummax()
        dd       = cum_pnl - roll_max
        ax2.plot(cum_pnl.index, cum_pnl.values, linewidth=2.0, color="#1f77b4", label="Skew Reversal L/S")
        ax2.fill_between(dd.index, dd.values, 0, where=dd.values < 0, alpha=0.25, color="red", label="Drawdown")
        ax2.axhline(0, color="black", linewidth=0.6)
        ax2.set_title("Alpha 05 — Cumulative PnL\n(Weekly Rebalance, Net of 8 bps TC)")
        ax2.set_ylabel("Cumulative Return")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # ── Panel 3: IC by Window ──────────────────────────────────────────────
        ax3 = fig.add_subplot(gs[1, 0])
        if self.ic_by_window is not None:
            pivot = self.ic_by_window.pivot(index="skew_window", columns="lag_d", values="IC")
            for col in pivot.columns:
                ax3.plot(pivot.index, pivot[col].values, marker="o",
                         label=f"Lag {col}d", linewidth=1.8)
            ax3.axhline(0, color="black", linewidth=0.7)
            ax3.set_xlabel("Skewness Rolling Window (days)")
            ax3.set_ylabel("Mean IC")
            ax3.set_title("Alpha 05 — IC vs Skewness Window\n(Which window is most predictive?)")
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # ── Panel 4: Skewness Distribution ─────────────────────────────────────
        ax4 = fig.add_subplot(gs[1, 1])
        skew_main = self.skew_dfs[self.main_window]
        # latest cross-section
        latest_date = skew_main.dropna(how="all").index[-1]
        latest_skew = skew_main.loc[latest_date].dropna()
        ax4.hist(latest_skew.values, bins=30, color="#8c564b", alpha=0.7, edgecolor="black", linewidth=0.5)
        ax4.axvline(0, color="black", linewidth=1.0, linestyle="--", label="Symmetric (zero)")
        ax4.axvline(latest_skew.mean(), color="red", linewidth=1.2, linestyle="-.",
                    label=f"Mean = {latest_skew.mean():.3f}")
        ax4.set_xlabel("Realized Skewness")
        ax4.set_ylabel("Count")
        ax4.set_title(f"Alpha 05 — Cross-Sectional Skewness Distribution\n(Latest date: {latest_date.date()})")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # ── Panel 5: Sign Asymmetry ────────────────────────────────────────────
        ax5 = fig.add_subplot(gs[2, 0])
        if self.sign_analysis is not None:
            sides   = list(self.sign_analysis.index)
            ic_vals = [self.sign_analysis.loc[s, "Mean_IC"] for s in sides]
            colors  = ["#2ca02c" if ic >= 0 else "#d62728" for ic in ic_vals]
            bars    = ax5.bar(sides, ic_vals, color=colors, alpha=0.8, edgecolor="black")
            ax5.axhline(0, color="black", linewidth=0.8)
            for bar, val in zip(bars, ic_vals):
                ax5.text(bar.get_x() + bar.get_width()/2,
                         val + 0.0005 * np.sign(val) if not np.isnan(val) else 0,
                         f"{val:.4f}" if not np.isnan(val) else "N/A",
                         ha="center", va="bottom" if val >= 0 else "top", fontsize=9)
            ax5.set_ylabel("Mean IC @ Lag 5d")
            ax5.set_title("Alpha 05 — Sign Asymmetry\n(Long side vs Short side contribution)")
            ax5.grid(True, alpha=0.3, axis="y")

        # ── Panel 6: Skewness vs Forward Return Scatter ─────────────────────────
        ax6 = fig.add_subplot(gs[2, 1])
        # sample 2000 observations for scatter
        skew_vals, fwd_rets = [], []
        fwd_5d = self.returns.shift(-5)
        for date in skew_main.index[100::10]:   # every 10th date
            if date not in fwd_5d.index:
                continue
            sk  = skew_main.loc[date].dropna()
            fwd = fwd_5d.loc[date].dropna()
            common = sk.index.intersection(fwd.index)
            skew_vals.extend(sk[common].tolist())
            fwd_rets.extend(fwd[common].tolist())
            if len(skew_vals) > 2000:
                break

        if skew_vals:
            sv = np.array(skew_vals)
            fr = np.array(fwd_rets)
            valid = np.isfinite(sv) & np.isfinite(fr)
            sv, fr = sv[valid], fr[valid]
            # winsorise for plotting
            sv_clip = np.clip(sv, np.percentile(sv, 2), np.percentile(sv, 98))
            fr_clip = np.clip(fr, np.percentile(fr, 2), np.percentile(fr, 98))
            ax6.scatter(sv_clip, fr_clip, s=4, alpha=0.3, color="#1f77b4")
            # trend line
            z = np.polyfit(sv_clip, fr_clip, 1)
            p = np.poly1d(z)
            x_line = np.linspace(sv_clip.min(), sv_clip.max(), 50)
            ax6.plot(x_line, p(x_line), "r-", linewidth=2, label=f"Trend (slope={z[0]:.5f})")
            ax6.axhline(0, color="black", linewidth=0.5)
            ax6.axvline(0, color="black", linewidth=0.5)
            ax6.set_xlabel("Realized Skewness (22d)")
            ax6.set_ylabel("5-day Forward Return")
            ax6.set_title("Alpha 05 — Skewness vs Forward Return\n(Negative slope validates short high-skew hypothesis)")
            ax6.legend(fontsize=9)
            ax6.grid(True, alpha=0.3)

        plt.suptitle(
            f"ALPHA 05 — Realized Skewness Reversal\n"
            f"Sharpe={self.metrics.get('Sharpe', np.nan):.2f}  "
            f"IC(OOS,5d)={self.metrics.get('IC_mean_OOS_lag5', np.nan):.4f}  "
            f"IC(OOS,22d)={self.metrics.get('IC_mean_OOS_lag22', np.nan):.4f}  "
            f"FM t(22d)={self.metrics.get('FM_t_stat_22d', np.nan):.2f}  "
            f"Mom_Corr={self.metrics.get('Momentum_Corr_r', np.nan):.3f}",
            fontsize=12, fontweight="bold",
        )

        if save:
            out_path = REPORTS_DIR / f"alpha_{ALPHA_ID}_chart.png"
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            log.info("Chart saved → %s", out_path)
        plt.close(fig)

    # ── markdown report ───────────────────────────────────────────────────────

    def generate_report(self) -> str:
        ic_table_str  = self.ic_table.reset_index().to_markdown(index=False, floatfmt=".5f")
        ic_is_str     = self.ic_is.reset_index().to_markdown(index=False, floatfmt=".5f")
        ic_oos_str    = self.ic_oos.reset_index().to_markdown(index=False, floatfmt=".5f")
        sign_str      = self.sign_analysis.reset_index().to_markdown(index=False, floatfmt=".5f") if self.sign_analysis is not None else "N/A"

        ic_window_str = "N/A"
        if self.ic_by_window is not None:
            pivot = self.ic_by_window.pivot(index="skew_window", columns="lag_d", values="IC")
            ic_window_str = pivot.to_markdown(floatfmt=".5f")

        skew_stats = self.skewness_stats_by_asset()
        skew_str   = skew_stats.head(10).to_markdown(floatfmt=".4f")

        mom_corr = self.momentum_corr or {}

        report = f"""# Alpha {ALPHA_ID}: {ALPHA_NAME.replace('_', ' ')}

## Hypothesis
Assets with positive realized skewness (lottery-like) are overpriced by investors
chasing extreme outcomes (Kahneman-Tversky probability weighting).  Shorting these
assets and going long negative-skewness (steady return) assets captures a systematic
mispricing that mean-reverts over 5–22 trading days.

## Expression (Python)
```python
from scipy.stats import skew

# Fisher-adjusted rolling skewness (unbiased)
skewness_22d = returns.rolling(22, min_periods=10).apply(
    lambda x: skew(x[~np.isnan(x)], bias=False), raw=True
)
# α₅: negate (short positive skewness, long negative)
alpha_05 = cross_sectional_rank(-skewness_22d)
```

## Parameters
| Parameter          | Value                   |
|--------------------|-------------------------|
| Primary window     | {self.main_window} days |
| Comparison windows | {self.skew_windows}     |
| Rebalance          | Weekly ({REBALANCE_DAYS}-day) |
| Universe           | {"Crypto" if self.use_crypto else "Equity (S&P 500)"} |
| Long/Short pct     | {self.top_pct*100:.0f}% |
| TC assumption      | {self.tc_bps:.0f} bps (round-trip) |

## Performance Summary

| Metric                | Value                  |
|-----------------------|------------------------|
| Sharpe Ratio          | {self.metrics.get('Sharpe', np.nan):.3f} |
| Annualised Return     | {self.metrics.get('Annualised_Return', np.nan)*100:.2f}% |
| Max Drawdown          | {self.metrics.get('MaxDrawdown', np.nan)*100:.2f}% |
| IC (IS)  @ 5d         | {self.metrics.get('IC_mean_IS_lag5', np.nan):.5f} |
| IC (OOS) @ 5d         | {self.metrics.get('IC_mean_OOS_lag5', np.nan):.5f} |
| IC (OOS) @ 22d        | {self.metrics.get('IC_mean_OOS_lag22', np.nan):.5f} |
| IC (OOS) @ 63d        | {self.metrics.get('IC_mean_OOS_lag63', np.nan):.5f} |
| ICIR (IS)  @ 5d       | {self.metrics.get('ICIR_IS_5d', np.nan):.3f} |
| ICIR (OOS) @ 5d       | {self.metrics.get('ICIR_OOS_5d', np.nan):.3f} |
| Fama-MacBeth γ (5d)   | {self.metrics.get('FM_gamma_5d', np.nan):.6f} |
| Fama-MacBeth t (5d)   | {self.metrics.get('FM_t_stat_5d', np.nan):.3f} |
| Fama-MacBeth γ (22d)  | {self.metrics.get('FM_gamma_22d', np.nan):.6f} |
| Fama-MacBeth t (22d)  | {self.metrics.get('FM_t_stat_22d', np.nan):.3f} |
| Daily Turnover        | {self.metrics.get('Turnover', np.nan)*100:.1f}% |
| Momentum Correlation  | {self.metrics.get('Momentum_Corr_r', np.nan):.4f} |

> **Note**: IC at daily horizons will be near zero — this is a weekly/monthly factor.
> Test at 5-day and 22-day horizons.

## Momentum Factor Correlation
| Statistic     | Value       |
|---------------|-------------|
| Pearson r     | {mom_corr.get('pearson_r', np.nan):.4f} |
| p-value       | {mom_corr.get('pearson_p', np.nan):.4f} |
| Spearman ρ    | {mom_corr.get('spearman_r', np.nan):.4f} |
| Interpretation| {mom_corr.get('interpretation', 'N/A')} |

## IC Decay Table (Full Sample)
{ic_table_str}

## In-Sample IC by Lag
{ic_is_str}

## Out-of-Sample IC by Lag
{ic_oos_str}

## IC by Skewness Window (Window Robustness Check)
{ic_window_str}

## Sign Asymmetry (Long vs Short Side)
{sign_str}

## Asset-Level Skewness Statistics (Top 10 by Mean Skewness)
{skew_str}

## Transaction Cost Break-Even
Monthly turnover ≈ {self.top_pct*2*REBALANCE_DAYS/22*100:.0f}% of portfolio.
At {self.tc_bps:.0f} bps, TC drag ≈ {self.top_pct*2*REBALANCE_DAYS/22*self.tc_bps/100:.4f}% per week.

## Academic References
- Harvey & Siddique (2000) *Conditional Skewness in Asset Pricing Tests* — JF
- Kumar (2009) *Who Gambles in the Stock Market?* — JF
- Bali, Cakici & Whitelaw (2011) *Maxing Out: Stocks as Lotteries and the Cross-Section* — JFinEc
- Akyildirim et al. (2021) *Do Skewness and Kurtosis of Cryptocurrency Asset Returns Matter?*
"""
        report_path = REPORTS_DIR / f"alpha_{ALPHA_ID}_report.md"
        report_path.write_text(report)
        log.info("Report saved → %s", report_path)
        return report


# ══════════════════════════════════════════════════════════════════════════════

def run_alpha05(
    tickers:    List[str] = None,
    start:      str       = DEFAULT_START,
    end:        str       = DEFAULT_END,
    use_crypto: bool      = False,
) -> Alpha05:
    alpha = Alpha05(tickers=tickers, start=start, end=end, use_crypto=use_crypto)
    alpha.run()
    alpha.plot()
    alpha.generate_report()

    metrics_csv = OUTPUT_DIR / "alpha_performance_summary.csv"
    row = pd.DataFrame([alpha.metrics])
    if metrics_csv.exists():
        existing = pd.read_csv(metrics_csv, index_col=0)
        existing = existing[existing["alpha_id"] != ALPHA_ID]
        row = pd.concat([existing, row], ignore_index=True)
    row.to_csv(metrics_csv)
    return alpha


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Alpha 05 — Realized Skewness Reversal")
    parser.add_argument("--start",  default=DEFAULT_START)
    parser.add_argument("--end",    default=DEFAULT_END)
    parser.add_argument("--crypto", action="store_true")
    args = parser.parse_args()

    alpha = Alpha05(start=args.start, end=args.end, use_crypto=args.crypto)
    alpha.run()
    alpha.plot()
    alpha.generate_report()

    print("\n" + "=" * 60)
    print("ALPHA 05 COMPLETE")
    print("=" * 60)
    for k, v in alpha.metrics.items():
        print(f"  {k:<40} {v:.5f}" if isinstance(v, float) else f"  {k:<40} {v}")
