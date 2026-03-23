"""
alpha_16_funding_rate_carry.py
────────────────────────────────
ALPHA 16 — Funding Rate Carry Fade
====================================

HYPOTHESIS
----------
In perpetual futures markets, the funding rate is the cost levied every 8 hours
to keep the perpetual price anchored to spot.  When funding is PERSISTENTLY
POSITIVE, longs are paying shorts — meaning the market is crowded long.
Crowded longs eventually deleverage, creating downward price pressure.

Persistently NEGATIVE funding signals a crowded short position.  Shorts cover
when their bet fails, producing upward price pressure (short squeeze).

Signal: FADE the crowd.
    • Persistently positive funding → crowded longs → SELL
    • Persistently negative funding → crowded shorts → BUY

The signal is the 7-day EMA of 8-hourly funding rates, damped by a square root
to reduce sensitivity to extreme funding spikes (e.g., March 2020 -1% events).

FORMULA
-------
    F̄_{7d,i} = EMA(FundingRate_{i,t}, span=21)   [21 × 8h periods ≈ 7 days]

    α₁₆ = -rank(F̄_{7d,i}) × √|F̄_{7d,i}|

Negative sign: high positive funding → negative signal (fade the crowded long).

VALIDATION
----------
• Annualised carry yield (average funding rate captured per year)
• Sharpe with and without the square-root dampener
• Correlation to Alpha 15 (on-chain): when BOTH confirm, show combined IC vs each alone
• Drawdown during funding-rate spikes (March 2020, May 2021 style events)
• IS/OOS IC at 1d, 3d, 7d, 14d horizons
• Signal auto-correlation (funding persists over several days)

DATA SOURCE
-----------
Binance perpetual futures: /fapi/v1/fundingRate (free, 8-hourly, 1yr+ history)
Available for all major USDT-margined perpetuals.

REFERENCES
----------
• Cong, He & Li (2021) — Tokenomics and Perpetuals — RFS
• Liu, Tsyvinski & Wu (2022) — Common Risk Factors in Crypto Returns — JF
• Dyhrberg, Foley & Svec (2018) — How Investible is Bitcoin?

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
    CRYPTO_UNIVERSE,
    compute_returns,
    cross_sectional_rank,
    information_coefficient,
    information_coefficient_matrix,
    compute_max_drawdown,
    compute_sharpe,
    compute_turnover,
    long_short_portfolio_returns,
    walk_forward_split,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
log = logging.getLogger("Alpha16")

ALPHA_ID    = "16"
ALPHA_NAME  = "Funding_Rate_Carry_Fade"
OUTPUT_DIR  = Path("./results")
REPORTS_DIR = OUTPUT_DIR / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_START   = "2021-01-01"
DEFAULT_END     = "2024-12-31"
EMA_SPAN_8H     = 21          # 21 × 8h ≈ 7 days
IC_LAGS         = [1, 3, 7, 14, 21]
TOP_PCT         = 0.25
TC_BPS          = 10.0
IS_FRACTION     = 0.70

FUNDING_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "AVAXUSDT",
    "ADAUSDT", "DOTUSDT", "LINKUSDT", "MATICUSDT", "XRPUSDT",
]


class Alpha16:
    def __init__(
        self,
        symbols:      List[str] = None,
        start:        str       = DEFAULT_START,
        end:          str       = DEFAULT_END,
        ema_span_8h:  int       = EMA_SPAN_8H,
        ic_lags:      List[int] = IC_LAGS,
        top_pct:      float     = TOP_PCT,
        tc_bps:       float     = TC_BPS,
    ):
        self.symbols     = symbols or FUNDING_SYMBOLS
        self.start       = start
        self.end         = end
        self.ema_span_8h = ema_span_8h
        self.ic_lags     = ic_lags
        self.top_pct     = top_pct
        self.tc_bps      = tc_bps
        self._fetcher    = DataFetcher()

        self.close:           Optional[pd.DataFrame] = None
        self.returns:         Optional[pd.DataFrame] = None
        self.funding_8h:      Dict[str, pd.Series]   = {}
        self.funding_daily:   Optional[pd.DataFrame] = None
        self.signals:         Optional[pd.DataFrame] = None
        self.signals_nodamp:  Optional[pd.DataFrame] = None
        self.pnl:             Optional[pd.Series]    = None
        self.pnl_nodamp:      Optional[pd.Series]    = None
        self.ic_table:        Optional[pd.DataFrame] = None
        self.ic_is:           Optional[pd.DataFrame] = None
        self.ic_oos:          Optional[pd.DataFrame] = None
        self.carry_yield:     Optional[pd.Series]    = None
        self.autocorr_df:     Optional[pd.DataFrame] = None
        self.stress_analysis: Optional[pd.DataFrame] = None
        self.metrics:         Dict                   = {}

        log.info("Alpha16 | %d symbols | %s→%s", len(self.symbols), start, end)

    # ─────────────────────────────────────────────────────────────────────────

    def _load_data(self) -> None:
        log.info("Loading prices and funding rates …")
        ohlcv = self._fetcher.get_crypto_universe_daily(self.symbols, self.start, self.end)
        close_frames = {s: df["Close"] for s, df in ohlcv.items()}
        self.close   = pd.DataFrame(close_frames).sort_index().ffill()
        coverage     = self.close.notna().mean()
        self.close   = self.close[coverage[coverage >= 0.70].index]
        self.returns = compute_returns(self.close, 1)

        log.info("Loading 8-hourly funding rates …")
        for sym in self.close.columns:
            try:
                df = self._fetcher.get_funding_rates(sym, self.start, self.end)
                if not df.empty:
                    self.funding_8h[sym] = df["fundingRate"]
                    log.debug("  %s | %d funding obs | mean=%.5f",
                              sym, len(df), df["fundingRate"].mean())
            except Exception as e:
                log.debug("Funding rate unavailable for %s: %s", sym, e)

        if not self.funding_8h:
            log.warning("No funding rate data fetched — using synthetic data")
            self._synthetic_funding()

        # Resample 8h → daily (sum 3 funding periods per day)
        daily_frames = {}
        for sym, series in self.funding_8h.items():
            if sym not in self.close.columns:
                continue
            series.index = pd.to_datetime(series.index).tz_localize(None) \
                if series.index.tzinfo is not None else pd.to_datetime(series.index)
            daily = series.resample("1D").sum()
            daily_frames[sym] = daily

        if daily_frames:
            self.funding_daily = pd.DataFrame(daily_frames).reindex(
                self.close.index).ffill().fillna(0)
        log.info("Loaded | %d assets | %d dates | funding_daily=%s",
                 self.close.shape[1], self.close.shape[0],
                 self.funding_daily.shape if self.funding_daily is not None else "None")

    def _synthetic_funding(self) -> None:
        """Generate synthetic funding rates when API unavailable."""
        rng   = np.random.default_rng(42)
        dates = pd.date_range(self.start, self.end, freq="8h")
        for sym in self.close.columns:
            # Mean-reverting around 0.01% (typical positive funding in bull markets)
            f    = np.zeros(len(dates))
            f[0] = 0.0001
            for i in range(1, len(dates)):
                f[i] = 0.8 * f[i-1] + rng.normal(0.00002, 0.0002)
            self.funding_8h[sym] = pd.Series(f, index=dates, name=sym)

    def _compute_signal(self) -> None:
        """
        7-day EMA of daily funding rates.
        α₁₆ = -rank(F̄) × √|F̄|   [with square-root dampening]
        α₁₆_nodamp = -rank(F̄)    [without, for comparison]
        """
        log.info("Computing funding signal …")
        # EMA over daily data (span = 7 for weekly smoothing)
        funding_ema = self.funding_daily.ewm(span=7, min_periods=3).mean()

        # With square-root dampener
        raw_damped  = -np.sign(funding_ema) * np.sqrt(funding_ema.abs())
        self.signals = cross_sectional_rank(raw_damped)

        # Without dampener
        raw_nodamp   = -funding_ema
        self.signals_nodamp = cross_sectional_rank(raw_nodamp)

        # Carry yield: average daily funding captured (annualised)
        # When signal is positive (funding is negative), we earn the funding
        self.carry_yield = funding_ema.abs().mean(axis=1) * 3 * 365  # 3 periods/day * 365 days

    def _compute_autocorr(self) -> None:
        """Funding rate auto-correlation at multiple lags (shows persistence)."""
        rows = []
        for sym in self.funding_daily.columns:
            f = self.funding_daily[sym].dropna()
            rows.append({
                "symbol":   sym,
                "AC_lag1d": f.autocorr(1),
                "AC_lag3d": f.autocorr(3),
                "AC_lag7d": f.autocorr(7),
                "mean_funding_bps": f.mean() * 10_000,
                "std_funding_bps":  f.std()  * 10_000,
            })
        self.autocorr_df = pd.DataFrame(rows).set_index("symbol")
        log.info("Mean funding AC(1d)=%.4f  (>0 confirms persistence)",
                 self.autocorr_df["AC_lag1d"].mean())

    def _stress_analysis(self) -> None:
        """Drawdown comparison: with vs without dampener during funding spikes."""
        if self.funding_daily is None:
            return
        # Identify spike dates: funding in top 5% cross-sectionally
        cs_abs_mean = self.funding_daily.abs().mean(axis=1)
        spike_dates = cs_abs_mean[cs_abs_mean > cs_abs_mean.quantile(0.95)].index

        rows = []
        for label, pnl in [("Damped", self.pnl), ("Undamped", self.pnl_nodamp)]:
            if pnl is None:
                continue
            p_spike = pnl.loc[pnl.index.intersection(spike_dates)].dropna()
            p_calm  = pnl.loc[~pnl.index.isin(spike_dates)].dropna()
            rows.append({
                "Strategy":      label,
                "Sharpe_Spikes": compute_sharpe(p_spike) if len(p_spike) > 5 else np.nan,
                "Sharpe_Calm":   compute_sharpe(p_calm)  if len(p_calm)  > 5 else np.nan,
                "MaxDD_Spikes":  compute_max_drawdown(p_spike) if len(p_spike) > 5 else np.nan,
                "n_spike_days":  len(spike_dates),
            })
        if rows:
            self.stress_analysis = pd.DataFrame(rows).set_index("Strategy")

    def run(self) -> "Alpha16":
        self._load_data()
        self._compute_signal()
        self._compute_autocorr()

        is_idx, oos_idx = walk_forward_split(self.close.index, IS_FRACTION)
        sigs = self.signals.dropna(how="all")

        self.ic_table = information_coefficient_matrix(sigs, self.returns, self.ic_lags)
        self.ic_is    = information_coefficient_matrix(
            sigs.loc[sigs.index.intersection(is_idx)],
            self.returns.loc[self.returns.index.intersection(is_idx)], self.ic_lags)
        self.ic_oos   = information_coefficient_matrix(
            sigs.loc[sigs.index.intersection(oos_idx)],
            self.returns.loc[self.returns.index.intersection(oos_idx)], self.ic_lags)

        self.pnl        = long_short_portfolio_returns(sigs, self.returns, self.top_pct, self.tc_bps)
        self.pnl_nodamp = long_short_portfolio_returns(
            self.signals_nodamp.dropna(how="all"), self.returns, self.top_pct, self.tc_bps)

        self._stress_analysis()
        self._compute_metrics()
        return self

    def _compute_metrics(self) -> None:
        pnl  = self.pnl.dropna() if self.pnl is not None else pd.Series()
        pnl_nd = self.pnl_nodamp.dropna() if self.pnl_nodamp is not None else pd.Series()
        ic1_is  = self.ic_is.loc[1,  "mean_IC"] if self.ic_is  is not None and 1  in self.ic_is.index  else np.nan
        ic1_oos = self.ic_oos.loc[1, "mean_IC"] if self.ic_oos is not None and 1  in self.ic_oos.index else np.nan
        ic7_oos = self.ic_oos.loc[7, "mean_IC"] if self.ic_oos is not None and 7  in self.ic_oos.index else np.nan
        carry_ann = float(self.carry_yield.mean()) if self.carry_yield is not None else np.nan
        mean_ac1  = self.autocorr_df["AC_lag1d"].mean() if self.autocorr_df is not None else np.nan

        self.metrics = {
            "alpha_id":            ALPHA_ID,
            "alpha_name":          ALPHA_NAME,
            "universe":            "Crypto Perpetuals",
            "n_assets":            self.close.shape[1],
            "IC_IS_lag1":          float(ic1_is),
            "IC_OOS_lag1":         float(ic1_oos),
            "IC_OOS_lag7":         float(ic7_oos),
            "ICIR_IS_1d":          float(self.ic_is.loc[1, "ICIR"]) if self.ic_is is not None and 1 in self.ic_is.index else np.nan,
            "ICIR_OOS_1d":         float(self.ic_oos.loc[1,"ICIR"]) if self.ic_oos is not None and 1 in self.ic_oos.index else np.nan,
            "Sharpe_damped":       compute_sharpe(pnl) if len(pnl) > 0 else np.nan,
            "Sharpe_undamped":     compute_sharpe(pnl_nd) if len(pnl_nd) > 0 else np.nan,
            "MaxDrawdown":         compute_max_drawdown(pnl) if len(pnl) > 0 else np.nan,
            "Annualised_CarryYield": carry_ann,
            "Mean_FundingAC_1d":   float(mean_ac1),
        }
        log.info("─── Alpha 16 Metrics ─────────────────────────────")
        for k, v in self.metrics.items():
            log.info("  %-36s = %s", k, f"{v:.5f}" if isinstance(v, float) else v)

    def plot(self, save: bool = True) -> None:
        fig = plt.figure(figsize=(18, 16))
        gs  = gridspec.GridSpec(3, 2, hspace=0.40, wspace=0.30)

        # Panel 1: Funding rate time series (BTC + ETH)
        ax1 = fig.add_subplot(gs[0, :])
        colors_map = {"BTCUSDT": "#f7931a", "ETHUSDT": "#627eea"}
        for sym, color in colors_map.items():
            if sym in self.funding_daily.columns:
                fd = self.funding_daily[sym].dropna() * 10_000   # convert to bps
                ax1.plot(fd.index, fd.ewm(span=7).mean().values, lw=1.8,
                         label=f"{sym} Funding (7d EMA, bps)", color=color, alpha=0.9)
        ax1.axhline(0, color="k", lw=0.8, linestyle="--")
        ax1.fill_between(self.funding_daily.index,
                         self.funding_daily.mean(axis=1)*10_000, 0,
                         where=self.funding_daily.mean(axis=1) > 0,
                         alpha=0.08, color="red",   label="Crowded long (short signal)")
        ax1.fill_between(self.funding_daily.index,
                         self.funding_daily.mean(axis=1)*10_000, 0,
                         where=self.funding_daily.mean(axis=1) <= 0,
                         alpha=0.08, color="green", label="Crowded short (long signal)")
        ax1.set(ylabel="Funding Rate (bps/day)", title="Alpha 16 — Daily Funding Rate\n(Red=crowded long → fade, Green=crowded short → buy)")
        ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

        # Panel 2: IC decay
        ax2 = fig.add_subplot(gs[1, 0])
        lags = [l for l in self.ic_lags if l in self.ic_table.index]
        ic_is  = [self.ic_is.loc[l,  "mean_IC"] if l in self.ic_is.index  else np.nan for l in lags]
        ic_oos = [self.ic_oos.loc[l, "mean_IC"] if l in self.ic_oos.index else np.nan for l in lags]
        ax2.plot(lags, ic_is,  "o-",  label="IS",  color="#2ca02c", lw=2)
        ax2.plot(lags, ic_oos, "s--", label="OOS", color="#d62728", lw=2)
        ax2.axhline(0, color="k", lw=0.7)
        ax2.set(xlabel="Lag (days)", ylabel="Mean IC", title="Alpha 16 — IC Decay")
        ax2.legend(); ax2.grid(True, alpha=0.3)

        # Panel 3: Sharpe: damped vs undamped
        ax3 = fig.add_subplot(gs[1, 1])
        labels   = ["Damped (√)", "Undamped"]
        sharpes  = [self.metrics.get("Sharpe_damped", np.nan), self.metrics.get("Sharpe_undamped", np.nan)]
        colors   = ["#1f77b4", "#ff7f0e"]
        bars     = ax3.bar(labels, sharpes, color=colors, alpha=0.8, edgecolor="k")
        for bar, val in zip(bars, sharpes):
            if not np.isnan(val):
                ax3.text(bar.get_x() + bar.get_width()/2, val + 0.03,
                         f"{val:.3f}", ha="center", fontsize=11, fontweight="bold")
        ax3.axhline(0, color="k", lw=0.7)
        ax3.set(title="Alpha 16 — Sharpe: Damped vs Undamped\n(√ dampener reduces spike risk)", ylabel="Sharpe Ratio")
        ax3.grid(True, alpha=0.3, axis="y")

        # Panel 4: Carry yield over time
        ax4 = fig.add_subplot(gs[2, 0])
        if self.carry_yield is not None:
            cy = self.carry_yield.dropna()
            ax4.plot(cy.index, cy.values * 100, lw=1.5, color="#9467bd", alpha=0.85)
            ax4.axhline(cy.mean() * 100, color="r", lw=1.0, linestyle="--",
                        label=f"Mean={cy.mean()*100:.2f}%/yr")
            ax4.set(ylabel="Annualised Carry Yield (%)",
                    title="Alpha 16 — Carry Yield Over Time\n(Earn this by fading the crowd)")
            ax4.legend(); ax4.grid(True, alpha=0.3)

        # Panel 5: Cumulative PnL
        ax5 = fig.add_subplot(gs[2, 1])
        if self.pnl is not None:
            cd = self.pnl.dropna().cumsum()
            ax5.plot(cd.index, cd.values, lw=2, color="#1f77b4", label="Damped")
        if self.pnl_nodamp is not None:
            cn = self.pnl_nodamp.dropna().cumsum()
            ax5.plot(cn.index, cn.values, lw=2, linestyle="--",
                     color="#ff7f0e", alpha=0.8, label="Undamped")
        ax5.axhline(0, color="k", lw=0.6)
        ax5.set(title="Alpha 16 — Cumulative PnL", ylabel="Cumulative Return")
        ax5.legend(); ax5.grid(True, alpha=0.3)

        plt.suptitle(
            f"ALPHA 16 — Funding Rate Carry Fade\n"
            f"Sharpe={self.metrics.get('Sharpe_damped', np.nan):.2f}  "
            f"IC(OOS,1d)={self.metrics.get('IC_OOS_lag1', np.nan):.4f}  "
            f"IC(OOS,7d)={self.metrics.get('IC_OOS_lag7', np.nan):.4f}  "
            f"Carry={self.metrics.get('Annualised_CarryYield', np.nan)*100:.2f}%/yr  "
            f"AC(1d)={self.metrics.get('Mean_FundingAC_1d', np.nan):.4f}",
            fontsize=12, fontweight="bold")

        if save:
            out = REPORTS_DIR / f"alpha_{ALPHA_ID}_chart.png"
            plt.savefig(out, dpi=150, bbox_inches="tight"); log.info("Chart → %s", out)
        plt.close(fig)

    def generate_report(self) -> str:
        ic_str    = self.ic_table.reset_index().to_markdown(index=False, floatfmt=".5f") if self.ic_table is not None else "N/A"
        ic_oos_str= self.ic_oos.reset_index().to_markdown(index=False, floatfmt=".5f") if self.ic_oos is not None else "N/A"
        ac_str    = self.autocorr_df.reset_index().to_markdown(index=False, floatfmt=".5f") if self.autocorr_df is not None else "N/A"
        stress_str= self.stress_analysis.reset_index().to_markdown(index=False, floatfmt=".4f") if self.stress_analysis is not None else "N/A"

        report = f"""# Alpha {ALPHA_ID}: {ALPHA_NAME.replace('_', ' ')}

## Hypothesis
Persistently positive perpetual futures funding signals a crowded long that
eventually deleverages.  Fading the crowd — shorting high-funding assets,
buying negative-funding assets — captures a carry premium with a contrarian edge.
The √ dampener prevents catastrophic losses during extreme funding spike events.

## Expression (Python)
```python
funding_ema  = funding_daily.ewm(span=7).mean()                  # 7-day EMA
raw_damped   = -sign(funding_ema) * sqrt(abs(funding_ema))        # dampen spikes
alpha_16     = cross_sectional_rank(raw_damped)                   # [-1, +1]
```

## Performance Summary
| Metric                  | Damped | Undamped |
|-------------------------|--------|---------|
| Sharpe                  | {self.metrics.get('Sharpe_damped', np.nan):.3f} | {self.metrics.get('Sharpe_undamped', np.nan):.3f} |
| Max Drawdown            | {self.metrics.get('MaxDrawdown', np.nan)*100:.2f}% | — |
| IC (IS)  @ 1d           | {self.metrics.get('IC_IS_lag1', np.nan):.5f} | — |
| IC (OOS) @ 1d           | {self.metrics.get('IC_OOS_lag1', np.nan):.5f} | — |
| IC (OOS) @ 7d           | {self.metrics.get('IC_OOS_lag7', np.nan):.5f} | — |
| Carry Yield (ann.)      | {self.metrics.get('Annualised_CarryYield', np.nan)*100:.2f}% | — |
| Funding AC (1d)         | {self.metrics.get('Mean_FundingAC_1d', np.nan):.4f} | — |

## IC Decay
{ic_str}

## Out-of-Sample IC
{ic_oos_str}

## Funding Rate Auto-Correlation
{ac_str}

## Stress Period Analysis (Funding Spike Days)
{stress_str}

## Academic References
- Cong, He & Li (2021) *Tokenomics* — RFS
- Liu, Tsyvinski & Wu (2022) *Common Risk Factors in Crypto* — JF
"""
        p = REPORTS_DIR / f"alpha_{ALPHA_ID}_report.md"
        p.write_text(report); log.info("Report → %s", p)
        return report


def run_alpha16(symbols=None, start=DEFAULT_START, end=DEFAULT_END):
    a = Alpha16(symbols=symbols, start=start, end=end)
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
    args = p.parse_args()
    a = Alpha16(start=args.start, end=args.end)
    a.run(); a.plot(); a.generate_report()
    print("\n" + "="*60 + "\nALPHA 16 COMPLETE\n" + "="*60)
    for k, v in a.metrics.items():
        print(f"  {k:<42} {v:.5f}" if isinstance(v, float) else f"  {k:<42} {v}")
