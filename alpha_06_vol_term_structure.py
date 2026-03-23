"""
alpha_06_vol_term_structure.py
───────────────────────────────
ALPHA 06 — Realized Volatility Term Structure (Vol Ratio Signal)
=================================================================

HYPOTHESIS
----------
The ratio of short-term realized volatility to long-term realized volatility
contains dual information:

1. As a STANDALONE ALPHA:
   When short-term vol >> long-term vol, the asset is in an agitated state —
   recent turbulence is likely mean-reverting.  Long low-vol-ratio (calm)
   assets, short high-vol-ratio (agitated) assets.
   α₆ = rank(RV_5d) - rank(RV_22d)   → short high ratio, long low ratio

2. As a REGIME PREDICTOR:
   High ratio → market is volatile → mean-reversion strategies work better
   Low ratio  → market is calm     → trend-following works better
   Cross-validates against HMM regime states (Alpha 09)

FORMULA
-------
    RV_{d,i} = sqrt( (252/d) × Σ_{k=1}^{d} r_{i,t-k}² )   (annualised realised vol)

    VolRatio_{i,t} = RV_{5d,i,t} / RV_{22d,i,t}

    α₆ = -rank(VolRatio_{i,t})     # short agitated (high ratio), long calm (low ratio)

ASSET CLASS
-----------
Primary: S&P 500 equities AND crypto (both markets tested)

REBALANCE FREQUENCY
-------------------
Weekly (5-day).  The vol-ratio signal changes at a medium pace; daily
rebalancing incurs excess TC with marginal IC improvement.

VALIDATION
----------
• Standalone IC at 5-day, 22-day horizons
• Correlation to HMM regime states (show it adds info beyond HMM)
• IC conditional on overall market vol level (VIX regime)
• Sharpe, Max Drawdown, Turnover
• Regime-conditional IC table (high/low market vol environment)

REFERENCES
----------
• Brandt, Kishore, Santa-Clara & Venkatesh (2010) — vol term structure
• Corsi (2009) — Heterogeneous Autoregressive (HAR) model
• Bollerslev, Tauchen & Zhou (2009) — Variance Risk Premium

Author : AI-Alpha-Factory
Version: 1.0.0
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional

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
log = logging.getLogger("Alpha06")

ALPHA_ID     = "06"
ALPHA_NAME   = "Vol_Term_Structure"
OUTPUT_DIR   = Path("./results")
REPORTS_DIR  = OUTPUT_DIR / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_START    = "2015-01-01"
DEFAULT_END      = "2024-12-31"
SHORT_VOL_WINDOW = 5
LONG_VOL_WINDOW  = 22
IC_LAGS          = [1, 5, 10, 22, 44]
TOP_PCT          = 0.20
TC_BPS           = 8.0
IS_FRACTION      = 0.70


class Alpha06:
    def __init__(
        self,
        tickers:          List[str] = None,
        start:            str       = DEFAULT_START,
        end:              str       = DEFAULT_END,
        short_vol_window: int       = SHORT_VOL_WINDOW,
        long_vol_window:  int       = LONG_VOL_WINDOW,
        ic_lags:          List[int] = IC_LAGS,
        top_pct:          float     = TOP_PCT,
        tc_bps:           float     = TC_BPS,
        use_crypto:       bool      = False,
    ):
        self.tickers          = tickers or (CRYPTO_UNIVERSE[:20] if use_crypto else SP500_TICKERS[:50])
        self.start            = start
        self.end              = end
        self.short_vol_window = short_vol_window
        self.long_vol_window  = long_vol_window
        self.ic_lags          = ic_lags
        self.top_pct          = top_pct
        self.tc_bps           = tc_bps
        self.use_crypto       = use_crypto
        self._fetcher         = DataFetcher()

        self.close:        Optional[pd.DataFrame] = None
        self.returns:      Optional[pd.DataFrame] = None
        self.rv_short:     Optional[pd.DataFrame] = None
        self.rv_long:      Optional[pd.DataFrame] = None
        self.vol_ratio:    Optional[pd.DataFrame] = None
        self.signals:      Optional[pd.DataFrame] = None
        self.pnl:          Optional[pd.Series]    = None
        self.ic_table:     Optional[pd.DataFrame] = None
        self.ic_is:        Optional[pd.DataFrame] = None
        self.ic_oos:       Optional[pd.DataFrame] = None
        self.vix:          Optional[pd.Series]    = None
        self.regime_ic:    Optional[pd.DataFrame] = None
        self.fm_result:    Dict                   = {}
        self.metrics:      Dict                   = {}

        log.info("Alpha06 | %d tickers | %s→%s", len(self.tickers), start, end)

    # ─────────────────────────────────────────────────────────────────────────

    def _load_data(self) -> None:
        log.info("Loading data …")
        if self.use_crypto:
            ohlcv = self._fetcher.get_crypto_universe_daily(self.tickers, self.start, self.end)
            close_frames = {s: df["Close"] for s, df in ohlcv.items()}
        else:
            ohlcv = self._fetcher.get_equity_ohlcv(self.tickers, self.start, self.end)
            close_frames = {t: df["Close"] for t, df in ohlcv.items() if not df.empty}

        self.close = pd.DataFrame(close_frames).sort_index().ffill()
        coverage   = self.close.notna().mean()
        self.close = self.close[coverage[coverage >= 0.80].index]
        self.returns = compute_returns(self.close, 1)

        if not self.use_crypto:
            try:
                self.vix = self._fetcher.get_vix(self.start, self.end)
            except Exception:
                self.vix = None

        log.info("Loaded | %d assets | %d dates", self.close.shape[1], self.close.shape[0])

    def _compute_realized_vol(self) -> None:
        """
        Annualised Realized Volatility:
            RV_{d} = sqrt(252/d × Σ r²)
        """
        log.info("Computing RV_%dd and RV_%dd …", self.short_vol_window, self.long_vol_window)
        r2 = self.returns ** 2

        rv_short_raw = r2.rolling(self.short_vol_window,  min_periods=3).mean()
        rv_long_raw  = r2.rolling(self.long_vol_window,   min_periods=10).mean()

        self.rv_short = np.sqrt(rv_short_raw * 252)
        self.rv_long  = np.sqrt(rv_long_raw  * 252)

    def _compute_signal(self) -> None:
        """
        Vol Ratio = RV_5d / RV_22d   (winsorised at 99th pct)
        α₆ = -rank(VolRatio)   →  short agitated, long calm
        """
        log.info("Computing vol ratio signal …")
        self.vol_ratio = (self.rv_short / self.rv_long.replace(0, np.nan)).clip(0, 5)
        self.vol_ratio = self.vol_ratio.apply(
            lambda col: col.clip(col.quantile(0.01), col.quantile(0.99)), axis=0
        )
        self.signals = cross_sectional_rank(-self.vol_ratio)

    def _compute_regime_conditional_ic(self) -> None:
        """
        Split dates into high-VIX (>20) and low-VIX (<20) regimes.
        Compute IC separately in each regime.
        """
        if self.vix is None:
            log.warning("VIX not available; skipping regime IC")
            return

        log.info("Computing regime-conditional IC …")
        fwd_5d  = self.returns.shift(-5)
        high_vix = self.vix[self.vix > 20].index
        low_vix  = self.vix[self.vix <= 20].index

        rows = []
        for regime_name, regime_dates in [("High VIX (>20)", high_vix), ("Low VIX (≤20)", low_vix)]:
            sigs = self.signals.loc[self.signals.index.intersection(regime_dates)]
            fwds = fwd_5d.loc[fwd_5d.index.intersection(regime_dates)]
            ic_stats = information_coefficient_matrix(sigs, fwds, [5])
            ic_val   = ic_stats.loc[5, "mean_IC"] if 5 in ic_stats.index else np.nan
            icir_val = ic_stats.loc[5, "ICIR"]    if 5 in ic_stats.index else np.nan
            rows.append({"Regime": regime_name, "mean_IC_5d": ic_val, "ICIR": icir_val,
                         "n_dates": len(regime_dates)})

        self.regime_ic = pd.DataFrame(rows).set_index("Regime")
        log.info("Regime IC:\n%s", self.regime_ic.to_string())

    def run(self) -> "Alpha06":
        self._load_data()
        self._compute_realized_vol()
        self._compute_signal()

        is_idx, oos_idx = walk_forward_split(self.close.index, IS_FRACTION)

        self.ic_table = information_coefficient_matrix(self.signals, self.returns, self.ic_lags)
        self.ic_is    = information_coefficient_matrix(
            self.signals.loc[self.signals.index.intersection(is_idx)],
            self.returns.loc[self.returns.index.intersection(is_idx)], self.ic_lags)
        self.ic_oos   = information_coefficient_matrix(
            self.signals.loc[self.signals.index.intersection(oos_idx)],
            self.returns.loc[self.returns.index.intersection(oos_idx)], self.ic_lags)

        self._compute_regime_conditional_ic()
        self.fm_result = fama_macbeth_regression(self.signals, self.returns, lag=5)

        self.pnl = long_short_portfolio_returns(
            self.signals, self.returns, self.top_pct, self.tc_bps)

        self._compute_metrics()
        return self

    def _compute_metrics(self) -> None:
        pnl = self.pnl.dropna()
        ic5_is  = self.ic_is.loc[5,  "mean_IC"] if 5  in self.ic_is.index  else np.nan
        ic5_oos = self.ic_oos.loc[5,  "mean_IC"] if 5  in self.ic_oos.index else np.nan
        ic22_oos= self.ic_oos.loc[22, "mean_IC"] if 22 in self.ic_oos.index else np.nan

        self.metrics = {
            "alpha_id":          ALPHA_ID,
            "alpha_name":        ALPHA_NAME,
            "universe":          "Crypto" if self.use_crypto else "Equity",
            "n_assets":          self.close.shape[1],
            "n_dates":           self.close.shape[0],
            "IC_mean_IS_lag5":   float(ic5_is),
            "IC_mean_OOS_lag5":  float(ic5_oos),
            "IC_mean_OOS_lag22": float(ic22_oos),
            "ICIR_IS_5d":        float(self.ic_is.loc[5,  "ICIR"]) if 5  in self.ic_is.index  else np.nan,
            "ICIR_OOS_5d":       float(self.ic_oos.loc[5, "ICIR"]) if 5  in self.ic_oos.index else np.nan,
            "FM_gamma_5d":       float(self.fm_result["gamma"]),
            "FM_t_stat_5d":      float(self.fm_result["t_stat"]),
            "Sharpe":            compute_sharpe(pnl),
            "MaxDrawdown":       compute_max_drawdown(pnl),
            "Annualised_Return": float(pnl.mean() * 252),
            "Turnover":          compute_turnover(self.signals),
        }
        log.info("─── Alpha 06 Metrics ────────────────────────────────")
        for k, v in self.metrics.items():
            log.info("  %-32s = %s", k, f"{v:.5f}" if isinstance(v, float) else v)

    def plot(self, save: bool = True) -> None:
        fig = plt.figure(figsize=(18, 14))
        gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.30)

        # Panel 1: IC decay
        ax1 = fig.add_subplot(gs[0, 0])
        lags_plot   = [l for l in self.ic_lags if l in self.ic_table.index]
        ic_is_vals  = [self.ic_is.loc[l,  "mean_IC"] if l in self.ic_is.index  else np.nan for l in lags_plot]
        ic_oos_vals = [self.ic_oos.loc[l, "mean_IC"] if l in self.ic_oos.index else np.nan for l in lags_plot]
        ax1.plot(lags_plot, ic_is_vals,  "o-",  label="IS",  color="#2ca02c", lw=2)
        ax1.plot(lags_plot, ic_oos_vals, "s--", label="OOS", color="#d62728", lw=2)
        ax1.axhline(0, color="k", lw=0.7)
        ax1.set(xlabel="Lag (days)", ylabel="Mean IC", title="Alpha 06 — IC Decay (Vol Ratio Signal)")
        ax1.legend(); ax1.grid(True, alpha=0.3)

        # Panel 2: Cumulative PnL
        ax2 = fig.add_subplot(gs[0, 1])
        cum = self.pnl.dropna().cumsum()
        ax2.plot(cum.index, cum.values, lw=2, color="#1f77b4")
        ax2.fill_between(cum.index, (cum - cum.cummax()).values, 0,
                         where=(cum - cum.cummax()).values < 0, alpha=0.3, color="red")
        ax2.axhline(0, color="k", lw=0.6)
        ax2.set(title="Alpha 06 — Cumulative PnL (Net of 8 bps TC)", ylabel="Cumulative Return")
        ax2.grid(True, alpha=0.3)

        # Panel 3: Vol-Ratio distribution snapshot
        ax3 = fig.add_subplot(gs[1, 0])
        latest = self.vol_ratio.dropna(how="all").iloc[-1].dropna()
        ax3.hist(latest.values, bins=30, color="#9467bd", alpha=0.75, edgecolor="k", lw=0.5)
        ax3.axvline(1.0, color="red", lw=1.5, linestyle="--", label="Ratio = 1 (neutral)")
        ax3.axvline(latest.mean(), color="green", lw=1.5, linestyle="-.", label=f"Mean={latest.mean():.2f}")
        ax3.set(xlabel="Vol Ratio (RV_5d / RV_22d)", ylabel="Count",
                title="Alpha 06 — Vol Ratio Cross-Section\n(Latest snapshot)")
        ax3.legend(); ax3.grid(True, alpha=0.3)

        # Panel 4: Regime-conditional IC
        ax4 = fig.add_subplot(gs[1, 1])
        if self.regime_ic is not None:
            regimes = list(self.regime_ic.index)
            ic_vals = [self.regime_ic.loc[r, "mean_IC_5d"] for r in regimes]
            colors  = ["#d62728" if "High" in r else "#2ca02c" for r in regimes]
            ax4.bar(regimes, ic_vals, color=colors, alpha=0.8, edgecolor="k")
            ax4.axhline(0, color="k", lw=0.8)
            for i, (r, v) in enumerate(zip(regimes, ic_vals)):
                ax4.text(i, v + 0.0003 * np.sign(v) if not np.isnan(v) else 0,
                         f"{v:.4f}" if not np.isnan(v) else "N/A",
                         ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
            ax4.set(ylabel="IC @ 5d lag", title="Alpha 06 — Regime-Conditional IC\n(High VIX vs Low VIX)")
            ax4.grid(True, alpha=0.3, axis="y")
        else:
            ax4.text(0.5, 0.5, "VIX data not available\n(crypto mode)", ha="center", va="center",
                     transform=ax4.transAxes, fontsize=12)

        plt.suptitle(
            f"ALPHA 06 — Vol Term Structure Signal\n"
            f"Sharpe={self.metrics.get('Sharpe', np.nan):.2f}  "
            f"IC(OOS,5d)={self.metrics.get('IC_mean_OOS_lag5', np.nan):.4f}  "
            f"FM t={self.metrics.get('FM_t_stat_5d', np.nan):.2f}",
            fontsize=13, fontweight="bold")

        if save:
            out = REPORTS_DIR / f"alpha_{ALPHA_ID}_chart.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            log.info("Chart → %s", out)
        plt.close(fig)

    def generate_report(self) -> str:
        ic_str    = self.ic_table.reset_index().to_markdown(index=False, floatfmt=".5f")
        ic_is_str = self.ic_is.reset_index().to_markdown(index=False, floatfmt=".5f")
        ic_oos_str= self.ic_oos.reset_index().to_markdown(index=False, floatfmt=".5f")
        reg_str   = self.regime_ic.reset_index().to_markdown(index=False, floatfmt=".5f") if self.regime_ic is not None else "N/A"

        report = f"""# Alpha {ALPHA_ID}: {ALPHA_NAME.replace('_', ' ')}

## Hypothesis
Short-term vol >> long-term vol signals agitation/turbulence that mean-reverts.
Short high-vol-ratio (turbulent) assets, long low-vol-ratio (calm) assets.
Also used as a regime indicator: high ratio → mean-reversion regime, low ratio → trend regime.

## Expression (Python)
```python
r2       = returns ** 2
rv_5d    = np.sqrt(r2.rolling(5,  min_periods=3).mean()  * 252)
rv_22d   = np.sqrt(r2.rolling(22, min_periods=10).mean() * 252)
vol_ratio = (rv_5d / rv_22d).clip(0, 5)
alpha_06  = cross_sectional_rank(-vol_ratio)   # short agitated, long calm
```

## Performance Summary
| Metric               | Value |
|----------------------|-------|
| Sharpe               | {self.metrics.get('Sharpe', np.nan):.3f} |
| Annualised Return    | {self.metrics.get('Annualised_Return', np.nan)*100:.2f}% |
| Max Drawdown         | {self.metrics.get('MaxDrawdown', np.nan)*100:.2f}% |
| IC (IS)  @ 5d        | {self.metrics.get('IC_mean_IS_lag5', np.nan):.5f} |
| IC (OOS) @ 5d        | {self.metrics.get('IC_mean_OOS_lag5', np.nan):.5f} |
| IC (OOS) @ 22d       | {self.metrics.get('IC_mean_OOS_lag22', np.nan):.5f} |
| ICIR (IS)  @ 5d      | {self.metrics.get('ICIR_IS_5d', np.nan):.3f} |
| ICIR (OOS) @ 5d      | {self.metrics.get('ICIR_OOS_5d', np.nan):.3f} |
| FM γ (5d)            | {self.metrics.get('FM_gamma_5d', np.nan):.6f} |
| FM t-stat (5d)       | {self.metrics.get('FM_t_stat_5d', np.nan):.3f} |
| Turnover             | {self.metrics.get('Turnover', np.nan)*100:.1f}% |

## IC Decay (Full Sample)
{ic_str}

## In-Sample IC
{ic_is_str}

## Out-of-Sample IC
{ic_oos_str}

## Regime-Conditional IC (VIX regime)
{reg_str}

## Academic References
- Corsi (2009) *A Simple Approximate Long-Memory Model of Realized Volatility* — JFinEc
- Brandt et al. (2010) *Parametric Portfolio Policies* — RFS
- Bollerslev, Tauchen & Zhou (2009) *Expected Stock Returns and Variance Risk Premia* — RFS
"""
        p = REPORTS_DIR / f"alpha_{ALPHA_ID}_report.md"
        p.write_text(report)
        log.info("Report → %s", p)
        return report


def run_alpha06(tickers=None, start=DEFAULT_START, end=DEFAULT_END, use_crypto=False):
    a = Alpha06(tickers=tickers, start=start, end=end, use_crypto=use_crypto)
    a.run(); a.plot(); a.generate_report()
    csv = OUTPUT_DIR / "alpha_performance_summary.csv"
    row = pd.DataFrame([a.metrics])
    if csv.exists():
        ex = pd.read_csv(csv, index_col=0)
        ex = ex[ex["alpha_id"] != ALPHA_ID]
        row = pd.concat([ex, row], ignore_index=True)
    row.to_csv(csv)
    return a


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--start",  default=DEFAULT_START)
    p.add_argument("--end",    default=DEFAULT_END)
    p.add_argument("--crypto", action="store_true")
    args = p.parse_args()
    a = Alpha06(start=args.start, end=args.end, use_crypto=args.crypto)
    a.run(); a.plot(); a.generate_report()
    print("\n" + "="*60 + "\nALPHA 06 COMPLETE\n" + "="*60)
    for k, v in a.metrics.items():
        print(f"  {k:<35} {v:.5f}" if isinstance(v, float) else f"  {k:<35} {v}")
