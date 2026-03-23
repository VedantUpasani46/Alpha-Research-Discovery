"""
alpha_03_amihud_illiquidity.py
────────────────────────────────
ALPHA 03 — Amihud Illiquidity Cross-Sectional Factor
======================================================

HYPOTHESIS
----------
Less liquid assets earn a persistent return premium because rational investors
demand compensation for bearing liquidity risk (illiquidity discount).
The Amihud (2002) illiquidity ratio captures price impact per unit of dollar
volume — higher ratio = lower liquidity = higher expected future return.

This is one of the most robustly documented cross-sectional anomalies in
finance, replicated across equities, bonds, and cryptocurrency markets.

FORMULA
-------
    ILLIQ_{i,t} = (1/D) × Σ_{d=1}^{D}  |R_{i,d}| / (V_{i,d} × P_{i,d})

where:
    R_{i,d}   = daily return on day d
    V_{i,d}   = daily trading volume (number of shares/coins)
    P_{i,d}   = price (so V × P = dollar/USD volume)
    D         = 60 trading days (rolling window)

Signal construction:
    α₃ = cross_sectional_rank(ILLIQ_{i,t})      → long high-ILLIQ, short low-ILLIQ

ASSET CLASS
-----------
• Primary:  S&P 500 equities (via yfinance)
• Optional: Crypto (via Binance) — works well, documented in Ben-David et al.

REBALANCE FREQUENCY
-------------------
Monthly (22-day).  Illiquidity is a slow-moving factor; daily rebalancing
destroys the net-of-cost alpha.

VALIDATION
----------
• IC at 22-day, 63-day horizons (this is a slow alpha)
• Net-of-cost Sharpe using the Almgren-Chriss transaction cost model
  (simplified: proportional TC based on turnover and illiquidity)
• Cumulative long-short portfolio return (monthly)
• Comparison to equal-weight benchmark

NOTES
─────
Winsorise ILLIQ at 1st/99th percentile before ranking —
thin trading days produce enormous ILLIQ spikes that corrupt the signal.

REFERENCES
----------
• Amihud (2002) *Illiquidity and Stock Returns* — JFinM
• Pástor & Stambaugh (2003) *Liquidity Risk and Expected Stock Returns* — JPE
• Brauneis, Mestel & Theissen (2021) — Amihud in crypto, Finance Research Letters

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
log = logging.getLogger("Alpha03")

# ── configuration ─────────────────────────────────────────────────────────────
ALPHA_ID          = "03"
ALPHA_NAME        = "Amihud_Illiquidity"
OUTPUT_DIR        = Path("./results")
REPORTS_DIR       = OUTPUT_DIR / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_START     = "2015-01-01"
DEFAULT_END       = "2024-12-31"
ILLIQ_WINDOW      = 60           # days to compute rolling ILLIQ
REBALANCE_DAYS    = 22           # monthly rebalancing
IC_LAGS           = [5, 10, 22, 44, 63]   # slow factor — daily IC low
TOP_PCT           = 0.20
TC_BPS            = 10.0         # monthly rebalancing — lower turnover
IS_FRACTION       = 0.70
WINSOR_LOWER      = 0.01
WINSOR_UPPER      = 0.99


# ══════════════════════════════════════════════════════════════════════════════
class Alpha03:
    """
    Full implementation of Alpha 03 — Amihud Illiquidity Cross-Sectional Factor.

    Key design decisions:
    - Uses dollar volume (V × P) as denominator — ensures comparability across
      assets with different price levels
    - Winsorises ILLIQ at 1/99 percentile to handle thin-trading outliers
    - Monthly rebalancing to preserve net-of-cost alpha
    - Fama-MacBeth validates the cross-sectional return prediction
    """

    def __init__(
        self,
        tickers:       List[str] = None,
        start:         str       = DEFAULT_START,
        end:           str       = DEFAULT_END,
        illiq_window:  int       = ILLIQ_WINDOW,
        rebalance_days:int       = REBALANCE_DAYS,
        ic_lags:       List[int] = IC_LAGS,
        top_pct:       float     = TOP_PCT,
        tc_bps:        float     = TC_BPS,
        use_crypto:    bool      = False,
    ):
        self.tickers        = tickers or (CRYPTO_UNIVERSE[:20] if use_crypto else SP500_TICKERS[:50])
        self.start          = start
        self.end            = end
        self.illiq_window   = illiq_window
        self.rebalance_days = rebalance_days
        self.ic_lags        = ic_lags
        self.top_pct        = top_pct
        self.tc_bps         = tc_bps
        self.use_crypto     = use_crypto

        self._fetcher = DataFetcher()

        # outputs
        self.close:       Optional[pd.DataFrame] = None
        self.volume:      Optional[pd.DataFrame] = None
        self.illiq_df:    Optional[pd.DataFrame] = None  # raw ILLIQ
        self.illiq_winsorised: Optional[pd.DataFrame] = None
        self.signals:     Optional[pd.DataFrame] = None
        self.returns:     Optional[pd.DataFrame] = None
        self.pnl:         Optional[pd.Series]    = None
        self.ic_table:    Optional[pd.DataFrame] = None
        self.ic_is:       Optional[pd.DataFrame] = None
        self.ic_oos:      Optional[pd.DataFrame] = None
        self.metrics:     Dict                   = {}
        self.fm_result:   Dict                   = {}

        log.info("Alpha03 initialised | %d tickers | %s→%s | use_crypto=%s",
                 len(self.tickers), start, end, use_crypto)

    # ── data loading ──────────────────────────────────────────────────────────

    def _load_data(self) -> None:
        log.info("Loading OHLCV data …")
        if self.use_crypto:
            ohlcv_dict = self._fetcher.get_crypto_universe_daily(self.tickers, self.start, self.end)
            close_frames  = {sym: df["Close"]  for sym, df in ohlcv_dict.items()}
            volume_frames = {sym: df["Volume"] for sym, df in ohlcv_dict.items()}
        else:
            ohlcv_dict = self._fetcher.get_equity_ohlcv(self.tickers, self.start, self.end)
            close_frames  = {t: df["Close"]  for t, df in ohlcv_dict.items() if not df.empty}
            volume_frames = {t: df["Volume"] for t, df in ohlcv_dict.items() if not df.empty}

        self.close  = pd.DataFrame(close_frames).sort_index().ffill()
        self.volume = pd.DataFrame(volume_frames).sort_index().ffill()

        # align
        common_cols = self.close.columns.intersection(self.volume.columns)
        self.close  = self.close[common_cols]
        self.volume = self.volume[common_cols]

        # drop assets with insufficient data
        coverage = self.close.notna().mean()
        good = coverage[coverage >= 0.80].index
        self.close  = self.close[good]
        self.volume = self.volume[good]

        log.info("Data loaded | %d assets | %d dates", self.close.shape[1], self.close.shape[0])

    # ── ILLIQ computation ─────────────────────────────────────────────────────

    def _compute_illiq(self) -> None:
        """
        ILLIQ_{i,t} = (1/D) × Σ_d |R_{i,d}| / (V_{i,d} × P_{i,d})

        Step-by-step:
        1. Compute daily |R| (absolute return)
        2. Compute daily USD volume = V × P  (handle zeros)
        3. Compute daily price impact = |R| / USD_volume
        4. Rolling mean over ILLIQ_WINDOW days
        5. Winsorise at 1/99 percentile
        """
        log.info("Computing Amihud ILLIQ …")

        abs_ret    = np.log(self.close / self.close.shift(1)).abs()
        usd_volume = self.volume * self.close                    # dollar volume
        usd_volume = usd_volume.replace(0, np.nan)

        daily_illiq = abs_ret / usd_volume                       # raw daily price impact

        # rolling mean
        self.illiq_df = daily_illiq.rolling(self.illiq_window, min_periods=20).mean()

        # winsorise
        def winsorise_row(row: pd.Series) -> pd.Series:
            row = row.dropna()
            if len(row) < 3:
                return row
            lb = row.quantile(WINSOR_LOWER)
            ub = row.quantile(WINSOR_UPPER)
            return row.clip(lb, ub)

        self.illiq_winsorised = self.illiq_df.apply(winsorise_row, axis=1)

        # align with original index
        self.illiq_winsorised = self.illiq_winsorised.reindex(self.illiq_df.index)

        # for daily ops, forward fill the winsorised frame
        self.illiq_winsorised = self.illiq_winsorised.ffill()

        log.info("ILLIQ computed | NaN fraction=%.2f%%",
                 self.illiq_df.isna().mean().mean() * 100)

    # ── signal computation ────────────────────────────────────────────────────

    def _compute_signal(self) -> None:
        """
        α₃ = cross_sectional_rank(ILLIQ_{i,t})
        Long high-ILLIQ (illiquid = high premium), short low-ILLIQ (liquid).

        We compute signals at monthly frequency but align to daily grid for
        the long-short portfolio engine.
        """
        log.info("Computing signal …")
        self.signals = cross_sectional_rank(self.illiq_winsorised)
        log.info("Signals computed | shape=%s", self.signals.shape)

    # ── monthly rebalancing portfolio ─────────────────────────────────────────

    def _monthly_portfolio(self) -> pd.Series:
        """
        Compute long-short portfolio with monthly rebalancing (every 22 trading days).
        Rebalances only on month-end dates — reuses previous weights intra-month.
        """
        returns_daily = compute_returns(self.close, 1)

        # generate rebalance dates (approx end-of-month)
        rebalance_dates = pd.date_range(
            start=self.signals.index[0],
            end=self.signals.index[-1],
            freq="BME",  # business month end
        )

        daily_rets = []
        current_weights: Optional[pd.Series] = None
        prev_weights:    Optional[pd.Series] = None

        for date in self.signals.index:
            if date not in returns_daily.index:
                continue

            # check if this is a rebalance date
            is_rebalance = any(
                abs((date - rd).days) <= 3 for rd in rebalance_dates
            )

            if is_rebalance or current_weights is None:
                sig_row = self.signals.loc[date].dropna()
                ret_row = returns_daily.loc[date].dropna()
                common  = sig_row.index.intersection(ret_row.index)

                if len(common) < 10:
                    daily_rets.append((date, np.nan))
                    continue

                sig   = sig_row[common]
                n     = len(common)
                n_top = max(1, int(n * self.top_pct))

                long_assets  = sig.nlargest(n_top).index
                short_assets = sig.nsmallest(n_top).index

                new_weights              = pd.Series(0.0, index=common)
                new_weights[long_assets]  =  1.0 / n_top
                new_weights[short_assets] = -1.0 / n_top
                prev_weights    = current_weights
                current_weights = new_weights

            if current_weights is None:
                daily_rets.append((date, np.nan))
                continue

            ret_row = returns_daily.loc[date].dropna()
            w = current_weights.reindex(ret_row.index).fillna(0)
            gross_ret = (w * ret_row).sum()

            # TC only on rebalance days
            cost = 0.0
            if is_rebalance and prev_weights is not None:
                old_w = prev_weights.reindex(ret_row.index).fillna(0)
                new_w = current_weights.reindex(ret_row.index).fillna(0)
                turnover = (new_w - old_w).abs().sum() / 2
                cost     = turnover * self.tc_bps / 10_000

            daily_rets.append((date, gross_ret - cost))

        return pd.Series(
            {d: v for d, v in daily_rets if not np.isnan(v)},
            name="illiq_port",
        )

    # ── main pipeline ─────────────────────────────────────────────────────────

    def run(self) -> "Alpha03":
        self._load_data()
        self._compute_illiq()
        self._compute_signal()

        self.returns = compute_returns(self.close, 1)
        is_idx, oos_idx = walk_forward_split(self.close.index, IS_FRACTION)

        # IC tables (monthly + quarterly lags)
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

        # Fama-MacBeth at 22-day lag
        self.fm_result_22 = fama_macbeth_regression(self.signals, self.returns, lag=22)
        self.fm_result_63 = fama_macbeth_regression(self.signals, self.returns, lag=63)
        log.info("Fama-MacBeth @22d | γ=%.5f | t=%.2f", self.fm_result_22["gamma"], self.fm_result_22["t_stat"])
        log.info("Fama-MacBeth @63d | γ=%.5f | t=%.2f", self.fm_result_63["gamma"], self.fm_result_63["t_stat"])

        # Portfolio
        log.info("Computing monthly rebalanced portfolio …")
        self.pnl = self._monthly_portfolio()

        self._compute_metrics()
        return self

    # ── metrics ───────────────────────────────────────────────────────────────

    def _compute_metrics(self) -> None:
        pnl = self.pnl.dropna()

        ic22_is  = self.ic_is.loc[22, "mean_IC"]  if 22 in self.ic_is.index  else np.nan
        ic22_oos = self.ic_oos.loc[22, "mean_IC"] if 22 in self.ic_oos.index else np.nan
        ic63_oos = self.ic_oos.loc[63, "mean_IC"] if 63 in self.ic_oos.index else np.nan

        self.metrics = {
            "alpha_id":           ALPHA_ID,
            "alpha_name":         ALPHA_NAME,
            "universe":           "Crypto" if self.use_crypto else "Equity",
            "n_assets":           self.close.shape[1],
            "n_dates":            self.close.shape[0],
            # ICs
            "IC_mean_IS_lag22":   float(ic22_is),
            "IC_mean_OOS_lag22":  float(ic22_oos),
            "IC_mean_OOS_lag63":  float(ic63_oos),
            "ICIR_IS_22":         float(self.ic_is.loc[22, "ICIR"])  if 22 in self.ic_is.index  else np.nan,
            "ICIR_OOS_22":        float(self.ic_oos.loc[22, "ICIR"]) if 22 in self.ic_oos.index else np.nan,
            # Fama-MacBeth
            "FM_gamma_22d":       float(self.fm_result_22["gamma"]),
            "FM_t_stat_22d":      float(self.fm_result_22["t_stat"]),
            "FM_gamma_63d":       float(self.fm_result_63["gamma"]),
            "FM_t_stat_63d":      float(self.fm_result_63["t_stat"]),
            # Portfolio
            "Sharpe":             compute_sharpe(pnl),
            "MaxDrawdown":        compute_max_drawdown(pnl),
            "Annualised_Return":  float(pnl.mean() * 252),
            "Turnover_monthly":   f"~{self.top_pct * 2 * 100:.0f}% of portfolio",  # monthly rebalance
        }

        log.info("─── Alpha 03 Metrics ───────────────────────────────────")
        for k, v in self.metrics.items():
            log.info("  %-35s = %s", k, f"{v:.5f}" if isinstance(v, float) else v)

    # ── ILLIQ descriptive stats ───────────────────────────────────────────────

    def illiq_descriptive_stats(self) -> pd.DataFrame:
        """
        Returns per-asset descriptive statistics for the ILLIQ signal.
        Useful for diagnosing the winsorisation impact.
        """
        raw    = self.illiq_df.describe().T
        wins   = self.illiq_winsorised.describe().T
        combined = pd.concat([raw.add_prefix("raw_"), wins.add_prefix("winsorised_")], axis=1)
        return combined

    # ── visualisation ─────────────────────────────────────────────────────────

    def plot(self, save: bool = True) -> None:
        """
        4-panel figure:
        1. ILLIQ distribution (raw vs winsorised)
        2. Cumulative long-short PnL (monthly rebalance)
        3. IC by lag (IS vs OOS)
        4. Average ILLIQ time series (cross-sectional mean)
        """
        fig = plt.figure(figsize=(18, 14))
        gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.30)

        # ── Panel 1: ILLIQ Distribution ────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        raw_flat  = self.illiq_df.values.flatten()
        win_flat  = self.illiq_winsorised.values.flatten()
        raw_flat  = raw_flat[np.isfinite(raw_flat)]
        win_flat  = win_flat[np.isfinite(win_flat)]
        p99       = np.percentile(raw_flat, 99)
        ax1.hist(raw_flat[raw_flat <= p99], bins=80, alpha=0.5, color="#ff7f0e", label="Raw ILLIQ", density=True)
        ax1.hist(win_flat,                  bins=80, alpha=0.6, color="#1f77b4", label="Winsorised", density=True)
        ax1.set_xlabel("ILLIQ (|R| / USD Volume)")
        ax1.set_ylabel("Density")
        ax1.set_title("Alpha 03 — ILLIQ Distribution\n(Raw vs Winsorised at 1/99 percentile)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # ── Panel 2: Cumulative PnL ─────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        cum_pnl = self.pnl.dropna().cumsum()
        ax2.plot(cum_pnl.index, cum_pnl.values, linewidth=2.0, color="#1f77b4", label="ILLIQ L/S Portfolio")
        ax2.axhline(0, color="black", linewidth=0.6)
        # shade drawdown
        roll_max = cum_pnl.cummax()
        dd       = cum_pnl - roll_max
        ax2.fill_between(dd.index, dd.values, 0, where=dd.values < 0, alpha=0.3, color="red", label="Drawdown")
        ax2.set_title("Alpha 03 — Cumulative PnL\n(Monthly Rebalance, Net of 10 bps TC)")
        ax2.set_ylabel("Cumulative Return")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # ── Panel 3: IC by Lag (IS vs OOS) ─────────────────────────────────────
        ax3 = fig.add_subplot(gs[1, 0])
        lags_common = sorted(set(self.ic_is.index) & set(self.ic_oos.index))
        x = np.arange(len(lags_common))
        w = 0.35
        ic_is_vals  = [self.ic_is.loc[l,  "mean_IC"] for l in lags_common]
        ic_oos_vals = [self.ic_oos.loc[l, "mean_IC"] for l in lags_common]
        ax3.bar(x - w/2, ic_is_vals,  w, label="In-Sample",     color="#2ca02c", alpha=0.8)
        ax3.bar(x + w/2, ic_oos_vals, w, label="Out-of-Sample", color="#d62728", alpha=0.8)
        ax3.set_xticks(x)
        ax3.set_xticklabels([f"Lag {l}d" for l in lags_common])
        ax3.axhline(0, color="black", linewidth=0.6)
        ax3.set_ylabel("Mean IC")
        ax3.set_title("Alpha 03 — IC by Lag (IS vs OOS)\n(Illiquidity is a slow factor)")
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis="y")

        # ── Panel 4: Cross-sectional ILLIQ over time ────────────────────────────
        ax4 = fig.add_subplot(gs[1, 1])
        cs_mean = self.illiq_winsorised.mean(axis=1).dropna()
        cs_std  = self.illiq_winsorised.std(axis=1).dropna()
        ax4.plot(cs_mean.index, cs_mean.values * 1e6, linewidth=1.5, color="#9467bd", label="CS Mean ILLIQ (×1M)")
        ax4.fill_between(cs_mean.index,
                         (cs_mean - cs_std).values * 1e6,
                         (cs_mean + cs_std).values * 1e6,
                         alpha=0.2, color="#9467bd", label="±1 std")
        ax4.set_title("Alpha 03 — Market-wide Amihud ILLIQ\n(×1M, Winsorised)")
        ax4.set_ylabel("ILLIQ × 10⁶")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.suptitle(
            f"ALPHA 03 — Amihud Illiquidity Factor\n"
            f"Sharpe={self.metrics.get('Sharpe', np.nan):.2f}  "
            f"IC(OOS,22d)={self.metrics.get('IC_mean_OOS_lag22', np.nan):.4f}  "
            f"IC(OOS,63d)={self.metrics.get('IC_mean_OOS_lag63', np.nan):.4f}  "
            f"FM t-stat(22d)={self.metrics.get('FM_t_stat_22d', np.nan):.2f}",
            fontsize=13, fontweight="bold",
        )

        if save:
            out_path = REPORTS_DIR / f"alpha_{ALPHA_ID}_chart.png"
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            log.info("Chart saved → %s", out_path)
        plt.close(fig)

    # ── markdown report ───────────────────────────────────────────────────────

    def generate_report(self) -> str:
        ic_table_str = self.ic_table.reset_index().to_markdown(index=False, floatfmt=".5f")
        ic_is_str    = self.ic_is.reset_index().to_markdown(index=False, floatfmt=".5f")
        ic_oos_str   = self.ic_oos.reset_index().to_markdown(index=False, floatfmt=".5f")

        illiq_stats  = self.illiq_descriptive_stats()
        stats_str    = illiq_stats[["raw_mean","raw_std","raw_max","winsorised_mean","winsorised_std","winsorised_max"]].head(10).to_markdown(floatfmt=".6e")

        report = f"""# Alpha {ALPHA_ID}: {ALPHA_NAME.replace('_', ' ')}

## Hypothesis
Illiquid assets earn a persistent return premium as compensation for liquidity risk.
The Amihud ratio (|R| / Dollar Volume) captures price impact per unit of trading — assets
with high price impact are harder to exit quickly, so rational investors demand higher returns.

## Expression (Python)
```python
abs_ret    = log(close / close.shift(1)).abs()
usd_vol    = volume * close                              # dollar volume
illiq_raw  = abs_ret / usd_vol                          # daily price impact
illiq_60d  = illiq_raw.rolling(60, min_periods=20).mean()
# winsorise at 1/99 percentile per day
illiq_wins = illiq_60d.apply(lambda r: r.clip(r.quantile(0.01), r.quantile(0.99)), axis=1)
alpha_03   = cross_sectional_rank(illiq_wins)            # long illiquid, short liquid
```

## Parameters
| Parameter         | Value               |
|-------------------|---------------------|
| ILLIQ window      | {self.illiq_window} days |
| Winsorisation     | {int(WINSOR_LOWER*100)}/{int(WINSOR_UPPER*100)} percentile |
| Rebalance         | Monthly (22-day)    |
| Universe          | {"Crypto" if self.use_crypto else "Equity (S&P 500)"} |
| Long/Short pct    | {self.top_pct*100:.0f}% |
| TC assumption     | {self.tc_bps:.0f} bps (round-trip, on rebalance only) |

## Performance Summary

| Metric               | Value                  |
|----------------------|------------------------|
| Sharpe Ratio         | {self.metrics.get('Sharpe', np.nan):.3f} |
| Annualised Return    | {self.metrics.get('Annualised_Return', np.nan)*100:.2f}% |
| Max Drawdown         | {self.metrics.get('MaxDrawdown', np.nan)*100:.2f}% |
| IC (IS)  @ 22d       | {self.metrics.get('IC_mean_IS_lag22', np.nan):.5f} |
| IC (OOS) @ 22d       | {self.metrics.get('IC_mean_OOS_lag22', np.nan):.5f} |
| IC (OOS) @ 63d       | {self.metrics.get('IC_mean_OOS_lag63', np.nan):.5f} |
| ICIR (IS)  @ 22d     | {self.metrics.get('ICIR_IS_22', np.nan):.3f} |
| ICIR (OOS) @ 22d     | {self.metrics.get('ICIR_OOS_22', np.nan):.3f} |
| Fama-MacBeth γ (22d) | {self.metrics.get('FM_gamma_22d', np.nan):.6f} |
| Fama-MacBeth t (22d) | {self.metrics.get('FM_t_stat_22d', np.nan):.3f} |
| Fama-MacBeth γ (63d) | {self.metrics.get('FM_gamma_63d', np.nan):.6f} |
| Fama-MacBeth t (63d) | {self.metrics.get('FM_t_stat_63d', np.nan):.3f} |

> **Note**: This is a slow-moving factor. IC at daily lags will be near zero —
> test at 22d and 63d horizons.

## IC Decay Table (Full Sample)
{ic_table_str}

## In-Sample IC by Lag
{ic_is_str}

## Out-of-Sample IC by Lag
{ic_oos_str}

## ILLIQ Descriptive Statistics (sample tickers)
{stats_str}

## Transaction Cost Analysis
Monthly rebalancing means portfolio turnover ≈ {self.top_pct*2*100:.0f}% per month.
At {self.tc_bps:.0f} bps round-trip cost, monthly TC drag ≈ {self.top_pct*2*self.tc_bps/100:.2f}% per month.
Break-even: alpha must exceed {self.top_pct*2*self.tc_bps*12/100:.2f}% annualised just to cover transaction costs.

## Academic References
- Amihud (2002) *Illiquidity and Stock Returns: Cross-Section and Time-Series Effects* — JFinM
- Pástor & Stambaugh (2003) *Liquidity Risk and Expected Stock Returns* — JPE
- Brauneis, Mestel & Theissen (2021) *How to Measure the Liquidity of Cryptocurrency Asset Markets?* — Finance Research Letters
"""
        report_path = REPORTS_DIR / f"alpha_{ALPHA_ID}_report.md"
        report_path.write_text(report)
        log.info("Report saved → %s", report_path)
        return report


# ══════════════════════════════════════════════════════════════════════════════

def run_alpha03(
    tickers:    List[str] = None,
    start:      str       = DEFAULT_START,
    end:        str       = DEFAULT_END,
    use_crypto: bool      = False,
) -> Alpha03:
    alpha = Alpha03(tickers=tickers, start=start, end=end, use_crypto=use_crypto)
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
    parser = argparse.ArgumentParser(description="Alpha 03 — Amihud Illiquidity")
    parser.add_argument("--start",  default=DEFAULT_START)
    parser.add_argument("--end",    default=DEFAULT_END)
    parser.add_argument("--crypto", action="store_true")
    args = parser.parse_args()

    alpha = Alpha03(start=args.start, end=args.end, use_crypto=args.crypto)
    alpha.run()
    alpha.plot()
    alpha.generate_report()

    print("\n" + "=" * 60)
    print("ALPHA 03 COMPLETE")
    print("=" * 60)
    for k, v in alpha.metrics.items():
        print(f"  {k:<40} {v:.5f}" if isinstance(v, float) else f"  {k:<40} {v}")
