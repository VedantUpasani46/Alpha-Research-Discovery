"""
alpha_14_residual_momentum.py
──────────────────────────────
ALPHA 14 — Residual Momentum (Idiosyncratic Alpha)
===================================================

HYPOTHESIS
----------
Raw price momentum (12-1 month return) contains two components:
  1. MARKET momentum: systematic factor driven by macro trends (mostly noise for stock-picking)
  2. IDIOSYNCRATIC momentum: firm/asset-specific trend uncorrelated with the market

Residual momentum — the cumulative residual from regressing daily returns on market
and sector returns — isolates the idiosyncratic component.  This:
  • Has a better Sharpe ratio than raw momentum
  • Avoids the January reversal effect (momentum crashes in January are due to the
    systematic component reversing, not the idiosyncratic component)
  • Provides genuinely incremental information to the Alpha 02 VPIN momentum signal

FORMULA
-------
    Step 1: Regress each asset's returns on market returns (rolling 126-day OLS):
        r_{i,t} = α_i + β_i × r_{m,t} + ε_{i,t}

    Step 2: Cumulate residuals over 126 days (6 months), skipping the last 5 days:
        ResidMom_{i,t} = Σ_{k=5}^{126} ε̂_{i,t-k}

    Step 3: Cross-sectional rank:
        α₁₄ = cross_sectional_rank(ResidMom_{i,t})

ASSET CLASS
-----------
S&P 500 equities.  The market return is the equal-weight cross-sectional mean.
(Also works on crypto basket using BTC as the market proxy.)

REBALANCE FREQUENCY
-------------------
Monthly (22-day).  This is a slow-horizon factor.

VALIDATION
----------
• IC at 22-day, 63-day horizons
• Sharpe comparison: residual momentum vs raw momentum vs equal-weight
• Fama-MacBeth regression coefficients and t-statistics
• Correlation to Alpha 02 (VPIN momentum): expect ~0.3–0.5 (partial overlap)
• Crisis behavior: show residual momentum has shallower drawdowns than raw momentum

REFERENCES
----------
• Blitz, Huij & Martens (2011) *Residual Momentum* — JEF
• Fama & French (1993) — 3-factor model (market beta removal)
• Daniel & Moskowitz (2016) — Momentum crashes — JFinEc

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
    information_coefficient_matrix,
    compute_max_drawdown,
    compute_sharpe,
    compute_turnover,
    long_short_portfolio_returns,
    fama_macbeth_regression,
    walk_forward_split,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
log = logging.getLogger("Alpha14")

ALPHA_ID    = "14"
ALPHA_NAME  = "Residual_Momentum"
OUTPUT_DIR  = Path("./results")
REPORTS_DIR = OUTPUT_DIR / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_START    = "2015-01-01"
DEFAULT_END      = "2024-12-31"
REGRESSION_WIN   = 126     # rolling OLS window (6 months)
CUMULATION_WIN   = 126     # residual cumulation window
SKIP_DAYS        = 5       # skip most recent 5 days (reversal avoidance)
IC_LAGS          = [5, 10, 22, 44, 63]
TOP_PCT          = 0.20
TC_BPS           = 8.0
IS_FRACTION      = 0.70


class ResidualMomentumCalculator:
    """
    Computes idiosyncratic (residual) momentum for a cross-section of assets.

    Algorithm (vectorised for efficiency):
    1. For each date t:
       a. Regress r_{i,d} ~ r_{m,d} over [t-REGRESSION_WIN, t] for each asset i
       b. Get the residuals ε̂_{i,d} = r_{i,d} - (α̂_i + β̂_i × r_{m,d})
       c. Sum residuals from [t-CUMULATION_WIN, t-SKIP_DAYS]
    2. This gives ResidMom_{i,t} for all i
    3. Cross-sectionally rank
    """

    def __init__(
        self,
        regression_win: int = REGRESSION_WIN,
        cumulation_win: int = CUMULATION_WIN,
        skip_days:      int = SKIP_DAYS,
    ):
        self.regression_win = regression_win
        self.cumulation_win = cumulation_win
        self.skip_days      = skip_days

    def compute(
        self,
        returns:    pd.DataFrame,
        market_ret: pd.Series,
    ) -> pd.DataFrame:
        """
        Computes residual momentum for all assets at each date.

        Parameters
        ----------
        returns    : (date × asset) daily return DataFrame
        market_ret : (date,) market return series

        Returns
        -------
        (date × asset) residual momentum DataFrame
        """
        log.info("Computing residual momentum (regression_win=%d, cumul_win=%d) …",
                 self.regression_win, self.cumulation_win)

        # Align
        common_idx = returns.index.intersection(market_ret.index)
        ret  = returns.loc[common_idx]
        mkt  = market_ret.loc[common_idx]

        n_dates  = len(ret)
        n_assets = len(ret.columns)

        # Pre-compute rolling OLS residuals for ALL assets simultaneously
        # Using vectorised pandas rolling approach:
        # β_i = Cov(r_i, r_m) / Var(r_m) via rolling windows

        # Step 1: Rolling covariance and variance
        rolling_cov = pd.DataFrame(index=ret.index, columns=ret.columns, dtype=float)
        rolling_var = mkt.rolling(self.regression_win, min_periods=30).var()

        for asset in ret.columns:
            pair = pd.concat([ret[asset], mkt], axis=1)
            pair.columns = ["asset", "mkt"]
            cov = pair.rolling(self.regression_win, min_periods=30).cov()
            if isinstance(cov, pd.DataFrame):
                cov_am = cov.xs("asset", level=1)["mkt"] if "asset" in cov.columns.get_level_values(1) else \
                         cov.unstack()["mkt"]["asset"] if hasattr(cov.unstack(), "__getitem__") else \
                         pair.rolling(self.regression_win, min_periods=30).apply(
                             lambda x: np.cov(x[:, 0], x[:, 1])[0, 1], raw=True)
            else:
                cov_am = pair.rolling(self.regression_win, min_periods=30).apply(
                    lambda x: np.cov(x[:, 0], x[:, 1])[0, 1] if len(x) > 2 else np.nan, raw=True)
            rolling_cov[asset] = cov_am

        # Compute rolling beta for all assets
        rolling_beta  = rolling_cov.divide(rolling_var.replace(0, np.nan), axis=0)
        rolling_alpha_mean = ret.rolling(self.regression_win, min_periods=30).mean()
        mkt_mean      = mkt.rolling(self.regression_win, min_periods=30).mean()
        rolling_alpha = rolling_alpha_mean.subtract(
            rolling_beta.multiply(mkt_mean, axis=0))

        # Step 2: Compute residuals
        fitted_returns = rolling_alpha.add(rolling_beta.multiply(mkt, axis=0))
        residuals      = ret - fitted_returns

        # Step 3: Cumulate residuals over [t-CUMULATION_WIN, t-SKIP_DAYS]
        # Using rolling sum on lagged residuals
        # ResidMom_t = sum(ε_{t-CUMULATION_WIN} ... ε_{t-SKIP_DAYS})
        resid_shifted = residuals.shift(self.skip_days)
        resid_mom = resid_shifted.rolling(
            self.cumulation_win - self.skip_days,
            min_periods=20
        ).sum()

        log.info("Residual momentum computed | NaN fraction=%.2f%%",
                 resid_mom.isna().mean().mean() * 100)
        return resid_mom


# ══════════════════════════════════════════════════════════════════════════════
class Alpha14:
    """
    Residual Momentum Alpha — idiosyncratic alpha with better drawdown profile.
    """

    def __init__(
        self,
        tickers:        List[str] = None,
        start:          str       = DEFAULT_START,
        end:            str       = DEFAULT_END,
        regression_win: int       = REGRESSION_WIN,
        cumulation_win: int       = CUMULATION_WIN,
        skip_days:      int       = SKIP_DAYS,
        ic_lags:        List[int] = IC_LAGS,
        top_pct:        float     = TOP_PCT,
        tc_bps:         float     = TC_BPS,
        use_crypto:     bool      = False,
    ):
        self.tickers        = tickers or (CRYPTO_UNIVERSE[:20] if use_crypto else SP500_TICKERS[:50])
        self.start          = start
        self.end            = end
        self.regression_win = regression_win
        self.cumulation_win = cumulation_win
        self.skip_days      = skip_days
        self.ic_lags        = ic_lags
        self.top_pct        = top_pct
        self.tc_bps         = tc_bps
        self.use_crypto     = use_crypto

        self._fetcher  = DataFetcher()
        self._calc     = ResidualMomentumCalculator(regression_win, cumulation_win, skip_days)

        self.close:          Optional[pd.DataFrame] = None
        self.returns:        Optional[pd.DataFrame] = None
        self.market_ret:     Optional[pd.Series]    = None
        self.resid_mom:      Optional[pd.DataFrame] = None
        self.raw_mom:        Optional[pd.DataFrame] = None
        self.signals:        Optional[pd.DataFrame] = None
        self.raw_signals:    Optional[pd.DataFrame] = None
        self.pnl:            Optional[pd.Series]    = None
        self.pnl_raw:        Optional[pd.Series]    = None
        self.ic_table:       Optional[pd.DataFrame] = None
        self.ic_is:          Optional[pd.DataFrame] = None
        self.ic_oos:         Optional[pd.DataFrame] = None
        self.ic_raw:         Optional[pd.DataFrame] = None
        self.mom_corr:       Optional[float]        = None
        self.fm_result:      Dict                   = {}
        self.metrics:        Dict                   = {}

        log.info("Alpha14 | %d tickers | %s→%s", len(self.tickers), start, end)

    def _load_data(self) -> None:
        log.info("Loading prices …")
        if self.use_crypto:
            ohlcv = self._fetcher.get_crypto_universe_daily(self.tickers, self.start, self.end)
            close_frames = {s: df["Close"] for s, df in ohlcv.items()}
        else:
            ohlcv = self._fetcher.get_equity_ohlcv(self.tickers, self.start, self.end)
            close_frames = {t: df["Close"] for t, df in ohlcv.items() if not df.empty}
        self.close   = pd.DataFrame(close_frames).sort_index().ffill()
        coverage     = self.close.notna().mean()
        self.close   = self.close[coverage[coverage >= 0.80].index]
        self.returns = compute_returns(self.close, 1)
        self.market_ret = self.returns.mean(axis=1)
        log.info("Loaded | %d assets | %d dates", self.close.shape[1], self.close.shape[0])

    def _compute_signals(self) -> None:
        # Residual momentum
        self.resid_mom = self._calc.compute(self.returns, self.market_ret)
        self.signals   = cross_sectional_rank(self.resid_mom)

        # Raw 12-1 month momentum for comparison
        price_126_ago = self.close.shift(self.cumulation_win + self.skip_days)
        price_skip    = self.close.shift(self.skip_days)
        self.raw_mom  = np.log(price_skip / price_126_ago)
        self.raw_signals = cross_sectional_rank(self.raw_mom)

        # Correlation
        flat_res = self.signals.values.flatten()
        flat_raw = self.raw_signals.reindex(self.signals.index).values.flatten()
        valid    = np.isfinite(flat_res) & np.isfinite(flat_raw)
        if valid.sum() > 20:
            r, _ = sp_stats.pearsonr(flat_res[valid], flat_raw[valid])
            self.mom_corr = float(r)
        log.info("Corr(residual_mom, raw_mom) = %.4f", self.mom_corr or np.nan)

    def run(self) -> "Alpha14":
        self._load_data()
        self._compute_signals()

        is_idx, oos_idx = walk_forward_split(self.close.index, IS_FRACTION)
        sigs_r = self.signals.dropna(how="all")
        sigs_w = self.raw_signals.dropna(how="all")

        self.ic_table = information_coefficient_matrix(sigs_r, self.returns, self.ic_lags)
        self.ic_is    = information_coefficient_matrix(
            sigs_r.loc[sigs_r.index.intersection(is_idx)],
            self.returns.loc[self.returns.index.intersection(is_idx)], self.ic_lags)
        self.ic_oos   = information_coefficient_matrix(
            sigs_r.loc[sigs_r.index.intersection(oos_idx)],
            self.returns.loc[self.returns.index.intersection(oos_idx)], self.ic_lags)
        self.ic_raw   = information_coefficient_matrix(sigs_w, self.returns, [22, 63])

        self.fm_result = fama_macbeth_regression(sigs_r, self.returns, lag=22)

        self.pnl     = long_short_portfolio_returns(sigs_r, self.returns, self.top_pct, self.tc_bps)
        self.pnl_raw = long_short_portfolio_returns(sigs_w, self.returns, self.top_pct, self.tc_bps)

        self._compute_metrics()
        return self

    def _compute_metrics(self) -> None:
        pr = self.pnl.dropna()     if self.pnl     is not None else pd.Series()
        pw = self.pnl_raw.dropna() if self.pnl_raw is not None else pd.Series()

        ic22_is  = self.ic_is.loc[22,  "mean_IC"] if self.ic_is  is not None and 22 in self.ic_is.index  else np.nan
        ic22_oos = self.ic_oos.loc[22, "mean_IC"] if self.ic_oos is not None and 22 in self.ic_oos.index else np.nan
        ic63_oos = self.ic_oos.loc[63, "mean_IC"] if self.ic_oos is not None and 63 in self.ic_oos.index else np.nan
        ic22_raw = self.ic_raw.loc[22, "mean_IC"] if self.ic_raw is not None and 22 in self.ic_raw.index else np.nan

        self.metrics = {
            "alpha_id":           ALPHA_ID,
            "alpha_name":         ALPHA_NAME,
            "universe":           "Crypto" if self.use_crypto else "Equity",
            "n_assets":           self.close.shape[1],
            "IC_IS_lag22":        float(ic22_is),
            "IC_OOS_lag22":       float(ic22_oos),
            "IC_OOS_lag63":       float(ic63_oos),
            "IC_raw_mom_lag22":   float(ic22_raw),
            "IC_resid_vs_raw":    float(ic22_oos - ic22_raw) if not np.isnan(ic22_oos + ic22_raw) else np.nan,
            "ICIR_IS_22d":        float(self.ic_is.loc[22, "ICIR"]) if self.ic_is is not None and 22 in self.ic_is.index else np.nan,
            "ICIR_OOS_22d":       float(self.ic_oos.loc[22,"ICIR"]) if self.ic_oos is not None and 22 in self.ic_oos.index else np.nan,
            "FM_gamma_22d":       float(self.fm_result.get("gamma", np.nan)),
            "FM_t_stat_22d":      float(self.fm_result.get("t_stat", np.nan)),
            "Sharpe_residual":    compute_sharpe(pr) if len(pr) > 0 else np.nan,
            "Sharpe_raw_mom":     compute_sharpe(pw) if len(pw) > 0 else np.nan,
            "MaxDD_residual":     compute_max_drawdown(pr) if len(pr) > 0 else np.nan,
            "MaxDD_raw_mom":      compute_max_drawdown(pw) if len(pw) > 0 else np.nan,
            "Mom_correlation":    float(self.mom_corr) if self.mom_corr is not None else np.nan,
        }
        log.info("─── Alpha 14 Metrics ─────────────────────────────")
        for k, v in self.metrics.items():
            log.info("  %-34s = %s", k, f"{v:.5f}" if isinstance(v, float) else v)

    def plot(self, save: bool = True) -> None:
        fig = plt.figure(figsize=(18, 14))
        gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.30)

        # Panel 1: IC decay (residual vs raw)
        ax1 = fig.add_subplot(gs[0, 0])
        lags = [l for l in self.ic_lags if l in self.ic_table.index]
        ic_r  = [self.ic_oos.loc[l, "mean_IC"] if l in self.ic_oos.index else np.nan for l in lags]
        ic_w  = [self.ic_raw.loc[l, "mean_IC"] if self.ic_raw is not None and l in self.ic_raw.index else np.nan for l in lags]
        ax1.plot(lags, ic_r, "o-",  label="Residual Mom (OOS)", color="#1f77b4", lw=2)
        ax1.plot(lags, ic_w, "s--", label="Raw Mom (OOS)",      color="#ff7f0e", lw=2)
        ax1.axhline(0, color="k", lw=0.7)
        ax1.set(xlabel="Lag (days)", ylabel="Mean IC",
                title="Alpha 14 — IC Decay: Residual vs Raw Momentum")
        ax1.legend(); ax1.grid(True, alpha=0.3)

        # Panel 2: Cumulative PnL comparison
        ax2 = fig.add_subplot(gs[0, 1])
        if self.pnl is not None:
            cr = self.pnl.dropna().cumsum()
            ax2.plot(cr.index, cr.values, lw=2, color="#1f77b4", label="Residual Momentum")
        if self.pnl_raw is not None:
            cw = self.pnl_raw.dropna().cumsum()
            ax2.plot(cw.index, cw.values, lw=2, linestyle="--", color="#ff7f0e",
                     alpha=0.8, label="Raw Momentum")
        ax2.axhline(0, color="k", lw=0.6)
        ax2.set(title="Alpha 14 — Cumulative PnL\n(Residual better drawdown profile)",
                ylabel="Cumulative Return")
        ax2.legend(); ax2.grid(True, alpha=0.3)

        # Panel 3: Max drawdown comparison (rolling 252-day)
        ax3 = fig.add_subplot(gs[1, 0])
        if self.pnl is not None and self.pnl_raw is not None:
            for name, pnl, color in [
                ("Residual", self.pnl, "#1f77b4"),
                ("Raw Mom",  self.pnl_raw, "#ff7f0e"),
            ]:
                cum = pnl.dropna().cumsum()
                dd  = cum - cum.rolling(252, min_periods=1).max()
                ax3.plot(dd.index, dd.values * 100, lw=1.5, label=f"{name} DD", color=color, alpha=0.85)
        ax3.axhline(0, color="k", lw=0.6)
        ax3.set(xlabel="Date", ylabel="Drawdown (%)",
                title="Alpha 14 — Rolling Drawdown\n(Residual = shallower crisis drawdowns)")
        ax3.legend(); ax3.grid(True, alpha=0.3)

        # Panel 4: Residual momentum vs raw correlation scatter
        ax4 = fig.add_subplot(gs[1, 1])
        if self.signals is not None and self.raw_signals is not None:
            flat_r = self.signals.values.flatten()
            flat_w = self.raw_signals.reindex(self.signals.index).values.flatten()
            valid  = np.isfinite(flat_r) & np.isfinite(flat_w)
            sample = np.random.default_rng(42).choice(valid.sum(), min(2000, valid.sum()), replace=False)
            ax4.scatter(flat_w[valid][sample], flat_r[valid][sample], s=3, alpha=0.2, color="#9467bd")
            z = np.polyfit(flat_w[valid][sample], flat_r[valid][sample], 1)
            x_line = np.linspace(flat_w[valid][sample].min(), flat_w[valid][sample].max(), 50)
            ax4.plot(x_line, np.poly1d(z)(x_line), "r-", lw=2,
                     label=f"r = {self.mom_corr:.3f}" if self.mom_corr else "trend")
            ax4.set(xlabel="Raw Momentum Signal", ylabel="Residual Momentum Signal",
                    title="Alpha 14 — Residual vs Raw Momentum\n(Partial overlap — they are NOT identical)")
            ax4.legend(); ax4.grid(True, alpha=0.3)

        plt.suptitle(
            f"ALPHA 14 — Residual Momentum\n"
            f"Sharpe_residual={self.metrics.get('Sharpe_residual', np.nan):.2f}  "
            f"Sharpe_raw={self.metrics.get('Sharpe_raw_mom', np.nan):.2f}  "
            f"IC(OOS,22d)={self.metrics.get('IC_OOS_lag22', np.nan):.4f}  "
            f"IC_lift={self.metrics.get('IC_resid_vs_raw', np.nan):+.4f}  "
            f"Mom_corr={self.metrics.get('Mom_correlation', np.nan):.3f}",
            fontsize=11, fontweight="bold")

        if save:
            out = REPORTS_DIR / f"alpha_{ALPHA_ID}_chart.png"
            plt.savefig(out, dpi=150, bbox_inches="tight"); log.info("Chart → %s", out)
        plt.close(fig)

    def generate_report(self) -> str:
        ic_str    = self.ic_table.reset_index().to_markdown(index=False, floatfmt=".5f") if self.ic_table is not None else "N/A"
        ic_is_str = self.ic_is.reset_index().to_markdown(index=False, floatfmt=".5f") if self.ic_is is not None else "N/A"
        ic_oos_str= self.ic_oos.reset_index().to_markdown(index=False, floatfmt=".5f") if self.ic_oos is not None else "N/A"
        ic_raw_str= self.ic_raw.reset_index().to_markdown(index=False, floatfmt=".5f") if self.ic_raw is not None else "N/A"

        report = f"""# Alpha {ALPHA_ID}: {ALPHA_NAME.replace('_', ' ')}

## Hypothesis
Idiosyncratic (residual) momentum — the component of price momentum NOT explained
by market/sector returns — is more persistent and has shallower drawdowns than raw
momentum.  The residual avoids January-effect crashes because those are driven by
the systematic component.

## Expression (Python)
```python
# Rolling OLS: regress asset returns on market returns
beta   = rolling_cov(r_asset, r_market) / rolling_var(r_market)
alpha  = r_asset.rolling(126).mean() - beta * r_market.rolling(126).mean()
resid  = r_asset - (alpha + beta * r_market)   # idiosyncratic returns
# Cumulate skipping last 5 days
resid_mom = resid.shift(5).rolling(121).sum()
alpha_14  = cross_sectional_rank(resid_mom)
```

## Performance Summary
| Metric                | Residual | Raw Momentum |
|-----------------------|----------|-------------|
| Sharpe                | {self.metrics.get('Sharpe_residual', np.nan):.3f} | {self.metrics.get('Sharpe_raw_mom', np.nan):.3f} |
| Max Drawdown          | {self.metrics.get('MaxDD_residual', np.nan)*100:.2f}% | {self.metrics.get('MaxDD_raw_mom', np.nan)*100:.2f}% |
| IC (IS)  @ 22d        | {self.metrics.get('IC_IS_lag22', np.nan):.5f} | {self.metrics.get('IC_raw_mom_lag22', np.nan):.5f} |
| IC (OOS) @ 22d        | {self.metrics.get('IC_OOS_lag22', np.nan):.5f} | — |
| IC (OOS) @ 63d        | {self.metrics.get('IC_OOS_lag63', np.nan):.5f} | — |
| IC lift vs raw        | {self.metrics.get('IC_resid_vs_raw', np.nan):+.5f} | — |
| ICIR (OOS, 22d)       | {self.metrics.get('ICIR_OOS_22d', np.nan):.3f} | — |
| FM γ (22d)            | {self.metrics.get('FM_gamma_22d', np.nan):.6f} | — |
| FM t-stat (22d)       | {self.metrics.get('FM_t_stat_22d', np.nan):.3f} | — |
| Momentum Correlation  | {self.metrics.get('Mom_correlation', np.nan):.4f} | — |

## IC Decay (Full Sample)
{ic_str}

## In-Sample IC
{ic_is_str}

## Out-of-Sample IC
{ic_oos_str}

## Raw Momentum IC (Comparison)
{ic_raw_str}

## Academic References
- Blitz, Huij & Martens (2011) *Residual Momentum* — JEF
- Fama & French (1993) — 3-Factor Model — JFinEc
- Daniel & Moskowitz (2016) — Momentum Crashes — JFinEc
"""
        p = REPORTS_DIR / f"alpha_{ALPHA_ID}_report.md"
        p.write_text(report); log.info("Report → %s", p)
        return report


def run_alpha14(tickers=None, start=DEFAULT_START, end=DEFAULT_END, use_crypto=False):
    a = Alpha14(tickers=tickers, start=start, end=end, use_crypto=use_crypto)
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
    p.add_argument("--start",  default=DEFAULT_START)
    p.add_argument("--end",    default=DEFAULT_END)
    p.add_argument("--crypto", action="store_true")
    args = p.parse_args()
    a = Alpha14(start=args.start, end=args.end, use_crypto=args.crypto)
    a.run(); a.plot(); a.generate_report()
    print("\n" + "="*60 + "\nALPHA 14 COMPLETE\n" + "="*60)
    for k, v in a.metrics.items():
        print(f"  {k:<40} {v:.5f}" if isinstance(v, float) else f"  {k:<40} {v}")
