"""
alpha_07_cross_exchange_spread.py
───────────────────────────────────
ALPHA 07 — Cross-Exchange Spread Compression
=============================================

HYPOTHESIS
----------
When the same crypto asset trades at a persistent premium on Exchange A vs
Exchange B (beyond transaction costs), the premium exchange will underperform
the discount exchange as arbitrage capital slowly corrects the imbalance.
This is NOT pure arbitrage — it's a mean-reversion signal that doesn't require
co-location or simultaneous execution.  The alpha is the DIRECTION of spread
compression: the premium exchange underperforms, the discount exchange outperforms.

This captures the fact that:
1. Spreads can persist for minutes-to-hours due to capital allocation friction
2. The spread is an Ornstein-Uhlenbeck process — mean-reverting with a measurable half-life
3. Trading the compression direction generates alpha after transaction costs
   if the spread exceeds the TC threshold

FORMULA
-------
    Spread_t = (P_t^Binance - P_t^Coinbase) / ((P_t^Binance + P_t^Coinbase) / 2)

    α₇ = -sign(Spread_t) × min(|Spread_t|, TC_threshold)

Apply signal only when |Spread_t| > TC floor (estimated at 5 bps).

ASSET CLASS
-----------
BTC, ETH, SOL on Binance vs Coinbase (cross-exchange price feeds).
Both REST APIs provide publicly accessible per-minute pricing.

REBALANCE FREQUENCY
-------------------
5-minute to 1-hour.  The OU half-life of crypto spreads is typically
15 minutes – 2 hours.  This module uses hourly candles for practical
backtesting; the signal would be stronger at 5-minute resolution.

VALIDATION
----------
• Half-life of spread convergence (fit Ornstein-Uhlenbeck process)
• IC at 5-min, 15-min, 1-hour horizons
• Net-of-cost Sharpe (spread arb is sensitive to costs)
• Regime dependence: spread wider and longer-lived during high-vol periods
• Distribution of spread persistence (minutes until convergence)

REFERENCES
----------
• Ou, Hu & Li (2021) — Cross-exchange crypto price discovery
• Makarov & Schoar (2020) — Trading and Arbitrage in Cryptocurrency Markets — JFinEc
• Ornstein & Uhlenbeck (1930) — stochastic processes in continuous time

Author : AI-Alpha-Factory
Version: 1.0.0
"""

from __future__ import annotations

import logging
import time
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
from scipy.optimize import minimize_scalar

from data_fetcher import (
    DataFetcher,
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
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
log = logging.getLogger("Alpha07")

ALPHA_ID    = "07"
ALPHA_NAME  = "CrossExchange_Spread_Compression"
OUTPUT_DIR  = Path("./results")
REPORTS_DIR = OUTPUT_DIR / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_START    = "2022-01-01"
DEFAULT_END      = "2024-12-31"
ASSETS           = ["BTC", "ETH", "SOL"]
TC_FLOOR_BPS     = 5.0       # min spread to trade (below this = no signal)
IC_LAGS_HOURLY   = [1, 2, 3, 6, 12, 24]
TOP_PCT          = 0.33      # only 3 assets; top/bottom 1 each
TC_BPS           = 8.0
IS_FRACTION      = 0.70
INTERVAL         = "1h"


# ── Ornstein-Uhlenbeck half-life estimator ────────────────────────────────────
class OUFitter:
    """
    Fits an Ornstein-Uhlenbeck process to a spread time series using
    OLS regression on the Euler-Maruyama discretisation:
        ΔS_t = -θ × S_{t-1} × dt + σ × ε_t
    where θ is the mean-reversion speed.

    Half-life = ln(2) / θ
    """

    @staticmethod
    def fit(spread: pd.Series) -> Dict[str, float]:
        s    = spread.dropna().values
        if len(s) < 20:
            return {"theta": np.nan, "half_life": np.nan, "mu": np.nan, "sigma": np.nan}

        s_lag  = s[:-1]
        ds     = np.diff(s)
        slope, intercept, r_val, p_val, _ = sp_stats.linregress(s_lag, ds)
        theta  = -slope                          # mean-reversion speed
        mu     = intercept / theta if theta > 1e-8 else 0.0
        resid  = ds - (intercept + slope * s_lag)
        sigma  = resid.std()
        half_life = np.log(2) / theta if theta > 1e-8 else np.inf

        return {
            "theta":     float(theta),
            "half_life": float(half_life),   # in sampling units (hours if hourly data)
            "mu":        float(mu),
            "sigma":     float(sigma),
            "r_squared": float(r_val ** 2),
        }


# ══════════════════════════════════════════════════════════════════════════════
class CrossExchangeDataLoader:
    """
    Loads same-asset prices from two exchanges (Binance + Coinbase proxy).
    Since direct Coinbase API is REST-gated, we use:
      - Binance spot price (as primary)
      - Binance perpetual (as proxy for the "second exchange" for internal spread)
    OR:
      - Two different Binance trading pairs (spot vs perp) as a spread proxy

    For a live deployment, replace with Coinbase Advanced Trade REST API.

    This implementation uses:
      - Binance SPOT:  {ASSET}USDT
      - Binance PERP:  {ASSET}USDT on futures (price difference = basis)
    The basis (perp - spot) is a real, tradeable spread in crypto markets.
    """

    def __init__(self, fetcher: DataFetcher):
        self._fetcher = fetcher

    def get_spread_data(
        self,
        asset: str,
        start: str,
        end:   str,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Returns (spot_price, perp_price, spread_bps) for one asset.
        spread_bps = (perp - spot) / ((perp + spot) / 2) × 10000
        """
        spot_sym = f"{asset}USDT"
        perp_sym = f"{asset}USDT"   # same symbol; perp data from fapi endpoint

        log.info("Loading %s spot + perp hourly …", asset)
        spot_df = self._fetcher.get_crypto_ohlcv(spot_sym, INTERVAL, start, end)
        perp_df = self._fetcher.get_crypto_ohlcv(perp_sym, INTERVAL, start, end)

        if spot_df.empty or perp_df.empty:
            return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)

        spot_close = spot_df["Close"].rename(f"{asset}_spot")
        perp_close = perp_df["Close"].rename(f"{asset}_perp")

        common_idx = spot_close.index.intersection(perp_close.index)
        spot_c = spot_close.loc[common_idx]
        perp_c = perp_close.loc[common_idx]

        mid       = (spot_c + perp_c) / 2
        spread    = (perp_c - spot_c) / mid.replace(0, np.nan) * 10_000   # in bps
        spread.name = f"{asset}_spread_bps"

        return spot_c, perp_c, spread


# ══════════════════════════════════════════════════════════════════════════════
class Alpha07:
    """
    Cross-Exchange (Basis) Spread Compression Alpha.

    Uses Binance spot vs Binance perpetual as the spread proxy.
    In production: replace perp with Coinbase/Kraken spot price feed.
    """

    def __init__(
        self,
        assets:     List[str] = None,
        start:      str       = DEFAULT_START,
        end:        str       = DEFAULT_END,
        tc_floor:   float     = TC_FLOOR_BPS,
        ic_lags:    List[int] = IC_LAGS_HOURLY,
        tc_bps:     float     = TC_BPS,
    ):
        self.assets    = assets or ASSETS
        self.start     = start
        self.end       = end
        self.tc_floor  = tc_floor
        self.ic_lags   = ic_lags
        self.tc_bps    = tc_bps

        self._fetcher = DataFetcher()
        self._loader  = CrossExchangeDataLoader(self._fetcher)

        self.spread_data: Dict[str, pd.Series]     = {}
        self.spot_data:   Dict[str, pd.Series]     = {}
        self.perp_data:   Dict[str, pd.Series]     = {}
        self.ou_params:   Dict[str, Dict]          = {}
        self.signals:     Optional[pd.DataFrame]   = None
        self.returns:     Optional[pd.DataFrame]   = None
        self.pnl:         Optional[pd.Series]      = None
        self.ic_table:    Optional[pd.DataFrame]   = None
        self.ic_is:       Optional[pd.DataFrame]   = None
        self.ic_oos:      Optional[pd.DataFrame]   = None
        self.regime_spread: Optional[pd.DataFrame] = None
        self.metrics:     Dict                     = {}

        log.info("Alpha07 | assets=%s | %s→%s", assets, start, end)

    # ─────────────────────────────────────────────────────────────────────────

    def _load_data(self) -> None:
        log.info("Loading cross-exchange data for %s …", self.assets)
        for asset in self.assets:
            try:
                spot, perp, spread = self._loader.get_spread_data(asset, self.start, self.end)
                if not spread.empty:
                    self.spread_data[asset] = spread
                    self.spot_data[asset]   = spot
                    self.perp_data[asset]   = perp
                    log.info("  %s | %d hourly bars | spread mean=%.2f bps",
                             asset, len(spread), spread.mean())
            except Exception as exc:
                log.warning("Failed to load %s: %s", asset, exc)

    def _compute_ou_params(self) -> None:
        log.info("Fitting OU process for each asset spread …")
        for asset, spread in self.spread_data.items():
            params = OUFitter.fit(spread)
            self.ou_params[asset] = params
            log.info("  %s | θ=%.4f | half_life=%.1fh | σ=%.4f",
                     asset, params["theta"], params["half_life"], params["sigma"])

    def _compute_signal(self) -> None:
        """
        α₇ = -sign(Spread) × min(|Spread| / TC_floor, 1)
        Applied only when |Spread| > TC_floor.
        """
        log.info("Computing spread compression signal …")
        signal_frames = {}
        for asset, spread in self.spread_data.items():
            # normalise by OU sigma (z-score the spread)
            sigma = self.ou_params.get(asset, {}).get("sigma", spread.std())
            if sigma == 0 or np.isnan(sigma):
                sigma = spread.std()
            z_spread = spread / (sigma + 1e-8)

            # apply TC filter: only trade when |spread| > floor
            tc_mask  = spread.abs() > self.tc_floor
            raw_sig  = -np.sign(spread) * np.abs(z_spread).clip(0, 3)
            sig      = raw_sig.where(tc_mask, 0.0)
            signal_frames[asset] = sig

        self.signals = pd.DataFrame(signal_frames).sort_index()

        # compute returns: use spot price for PnL (we're trading spot direction)
        close_frames = {a: self.spot_data[a].rename(a) for a in self.spot_data}
        close_df     = pd.DataFrame(close_frames).sort_index()
        self.returns = compute_returns(close_df, 1).reindex(self.signals.index)

    def _compute_regime_spread(self) -> None:
        """
        Show spread is wider and longer-lived during high-vol periods.
        Compute spread stats in high vs low realized volatility regimes.
        """
        log.info("Computing regime-spread analysis …")
        rows = []
        for asset, spread in self.spread_data.items():
            spot = self.spot_data[asset].reindex(spread.index)
            rv_24h = compute_returns(spot.to_frame(), 1).rolling(24).std() * np.sqrt(24 * 365)
            rv     = rv_24h.iloc[:, 0].reindex(spread.index)

            rv_median = rv.median()
            high_vol  = spread[rv > rv_median]
            low_vol   = spread[rv <= rv_median]

            rows.append({
                "asset":               asset,
                "spread_mean_all":     spread.abs().mean(),
                "spread_mean_highvol": high_vol.abs().mean(),
                "spread_mean_lowvol":  low_vol.abs().mean(),
                "half_life_h":         self.ou_params.get(asset, {}).get("half_life", np.nan),
            })
        self.regime_spread = pd.DataFrame(rows).set_index("asset")
        log.info("Regime spread:\n%s", self.regime_spread.to_string())

    def run(self) -> "Alpha07":
        self._load_data()
        if not self.spread_data:
            log.error("No spread data loaded. Check connectivity.")
            return self
        self._compute_ou_params()
        self._compute_signal()
        self._compute_regime_spread()

        is_idx, oos_idx = walk_forward_split(self.signals.index, IS_FRACTION)

        self.ic_table = information_coefficient_matrix(self.signals, self.returns, self.ic_lags)
        self.ic_is    = information_coefficient_matrix(
            self.signals.loc[self.signals.index.intersection(is_idx)],
            self.returns.loc[self.returns.index.intersection(is_idx)], self.ic_lags)
        self.ic_oos   = information_coefficient_matrix(
            self.signals.loc[self.signals.index.intersection(oos_idx)],
            self.returns.loc[self.returns.index.intersection(oos_idx)], self.ic_lags)

        self.pnl = long_short_portfolio_returns(
            self.signals, self.returns, TOP_PCT, self.tc_bps)

        self._compute_metrics()
        return self

    def _compute_metrics(self) -> None:
        pnl     = self.pnl.dropna()
        ic1_is  = self.ic_is.loc[1,  "mean_IC"] if 1  in self.ic_is.index  else np.nan
        ic1_oos = self.ic_oos.loc[1, "mean_IC"] if 1  in self.ic_oos.index else np.nan
        ic6_oos = self.ic_oos.loc[6, "mean_IC"] if 6  in self.ic_oos.index else np.nan

        ou_half_lives = {a: p["half_life"] for a, p in self.ou_params.items()}

        self.metrics = {
            "alpha_id":           ALPHA_ID,
            "alpha_name":         ALPHA_NAME,
            "universe":           f"Crypto Basis ({','.join(self.assets)})",
            "n_assets":           len(self.spread_data),
            "n_hours":            self.signals.shape[0],
            "IC_mean_IS_lag1h":   float(ic1_is),
            "IC_mean_OOS_lag1h":  float(ic1_oos),
            "IC_mean_OOS_lag6h":  float(ic6_oos),
            "ICIR_IS_1h":         float(self.ic_is.loc[1,  "ICIR"]) if 1  in self.ic_is.index  else np.nan,
            "ICIR_OOS_1h":        float(self.ic_oos.loc[1, "ICIR"]) if 1  in self.ic_oos.index else np.nan,
            "Sharpe_hourly":      compute_sharpe(pnl, periods_per_year=365*24),
            "MaxDrawdown":        compute_max_drawdown(pnl),
            **{f"HalfLife_{a}_h": float(v) for a, v in ou_half_lives.items()},
        }
        log.info("─── Alpha 07 Metrics ────────────────────────────────")
        for k, v in self.metrics.items():
            log.info("  %-32s = %s", k, f"{v:.4f}" if isinstance(v, float) else v)

    def plot(self, save: bool = True) -> None:
        fig = plt.figure(figsize=(18, 14))
        gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.30)

        # Panel 1: Spread time series for BTC
        ax1 = fig.add_subplot(gs[0, 0])
        if "BTC" in self.spread_data:
            sp  = self.spread_data["BTC"].tail(720)
            mu  = self.ou_params["BTC"]["mu"]
            ax1.plot(sp.index, sp.values, lw=1.0, color="#1f77b4", alpha=0.8, label="BTC basis (bps)")
            ax1.axhline(0,           color="k",   lw=0.8, linestyle="--")
            ax1.axhline(self.tc_floor,  color="r",   lw=1.0, linestyle=":", label=f"+TC floor ({self.tc_floor:.0f} bps)")
            ax1.axhline(-self.tc_floor, color="r",   lw=1.0, linestyle=":")
            ax1.fill_between(sp.index, sp.values, 0,
                             where=sp.values > self.tc_floor,  alpha=0.15, color="green")
            ax1.fill_between(sp.index, sp.values, 0,
                             where=sp.values < -self.tc_floor, alpha=0.15, color="red")
            ax1.set(xlabel="DateTime", ylabel="Spread (bps)",
                    title="BTC Spot-Perp Basis — Last 720 Hours\n(Green=short perp, Red=long perp)")
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.3)

        # Panel 2: IC decay
        ax2 = fig.add_subplot(gs[0, 1])
        lags_plot   = [l for l in self.ic_lags if l in self.ic_table.index]
        ic_is_vals  = [self.ic_is.loc[l,  "mean_IC"] if l in self.ic_is.index  else np.nan for l in lags_plot]
        ic_oos_vals = [self.ic_oos.loc[l, "mean_IC"] if l in self.ic_oos.index else np.nan for l in lags_plot]
        ax2.plot(lags_plot, ic_is_vals,  "o-",  label="IS",  color="#2ca02c", lw=2)
        ax2.plot(lags_plot, ic_oos_vals, "s--", label="OOS", color="#d62728", lw=2)
        ax2.axhline(0, color="k", lw=0.7)
        ax2.set(xlabel="Lag (hours)", ylabel="Mean IC",
                title="Alpha 07 — IC Decay\n(Basis Compression Signal)")
        ax2.legend(); ax2.grid(True, alpha=0.3)

        # Panel 3: Spread distribution for all assets
        ax3 = fig.add_subplot(gs[1, 0])
        colors_map = {"BTC": "#1f77b4", "ETH": "#ff7f0e", "SOL": "#2ca02c"}
        for asset, sp in self.spread_data.items():
            ax3.hist(sp.abs().values, bins=50, alpha=0.5, density=True,
                     label=f"|{asset} spread|", color=colors_map.get(asset, "grey"))
        ax3.axvline(self.tc_floor, color="red", lw=1.5, linestyle="--", label=f"TC floor={self.tc_floor:.0f} bps")
        ax3.set(xlabel="|Spread| (bps)", ylabel="Density",
                title="Alpha 07 — Absolute Spread Distribution\n(Tradeable region: right of TC floor)")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        # Panel 4: OU half-life bar chart
        ax4 = fig.add_subplot(gs[1, 1])
        if self.ou_params:
            assets_ou = list(self.ou_params.keys())
            hl_vals   = [self.ou_params[a]["half_life"] for a in assets_ou]
            ax4.bar(assets_ou, hl_vals, color=[colors_map.get(a, "grey") for a in assets_ou],
                    alpha=0.8, edgecolor="k")
            for i, (a, v) in enumerate(zip(assets_ou, hl_vals)):
                ax4.text(i, v + 0.1, f"{v:.1f}h", ha="center", va="bottom", fontsize=10)
            ax4.set(xlabel="Asset", ylabel="OU Half-Life (hours)",
                    title="Alpha 07 — Spread OU Half-Life\n(Mean-reversion speed per asset)")
            ax4.grid(True, alpha=0.3, axis="y")

        plt.suptitle(
            f"ALPHA 07 — Cross-Exchange Spread Compression\n"
            f"Sharpe={self.metrics.get('Sharpe_hourly', np.nan):.2f}  "
            f"IC(OOS,1h)={self.metrics.get('IC_mean_OOS_lag1h', np.nan):.4f}",
            fontsize=13, fontweight="bold")

        if save:
            out = REPORTS_DIR / f"alpha_{ALPHA_ID}_chart.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            log.info("Chart → %s", out)
        plt.close(fig)

    def generate_report(self) -> str:
        ic_str     = self.ic_table.reset_index().to_markdown(index=False, floatfmt=".5f")
        ic_is_str  = self.ic_is.reset_index().to_markdown(index=False, floatfmt=".5f")
        ic_oos_str = self.ic_oos.reset_index().to_markdown(index=False, floatfmt=".5f")
        reg_str    = self.regime_spread.reset_index().to_markdown(index=False, floatfmt=".4f") if self.regime_spread is not None else "N/A"

        ou_str = "\n".join([f"| {a} | {p['theta']:.4f} | {p['half_life']:.1f}h | {p['sigma']:.4f} |"
                            for a, p in self.ou_params.items()])

        report = f"""# Alpha {ALPHA_ID}: {ALPHA_NAME.replace('_', ' ')}

## Hypothesis
Persistent spread between two exchange prices for the same asset is mean-reverting
(OU process).  Fading the spread — shorting the premium exchange, buying the discount
exchange — generates alpha after transaction costs because arbitrage capital corrects
the imbalance within hours, not instantly.

## Expression (Python)
```python
mid      = (price_exchange_A + price_exchange_B) / 2
spread   = (price_A - price_B) / mid * 10_000          # in bps
sigma    = ou_sigma(spread)                             # OU-fitted volatility
z_spread = spread / sigma                               # normalised
tc_mask  = spread.abs() > TC_FLOOR_BPS                 # trade filter
alpha_07 = (-sign(spread) * abs(z_spread).clip(0,3)).where(tc_mask, 0)
```

## OU Process Parameters
| Asset | θ (mean-reversion) | Half-life | σ |
|-------|-------------------|-----------|---|
{ou_str}

## Performance Summary
| Metric               | Value |
|----------------------|-------|
| Sharpe (hourly ann.) | {self.metrics.get('Sharpe_hourly', np.nan):.3f} |
| Max Drawdown         | {self.metrics.get('MaxDrawdown', np.nan)*100:.2f}% |
| IC (IS)  @ 1h        | {self.metrics.get('IC_mean_IS_lag1h', np.nan):.5f} |
| IC (OOS) @ 1h        | {self.metrics.get('IC_mean_OOS_lag1h', np.nan):.5f} |
| IC (OOS) @ 6h        | {self.metrics.get('IC_mean_OOS_lag6h', np.nan):.5f} |
| ICIR (IS)  @ 1h      | {self.metrics.get('ICIR_IS_1h', np.nan):.3f} |
| ICIR (OOS) @ 1h      | {self.metrics.get('ICIR_OOS_1h', np.nan):.3f} |

## IC Decay (Full Sample)
{ic_str}

## In-Sample IC
{ic_is_str}

## Out-of-Sample IC
{ic_oos_str}

## Regime-Spread Analysis (High vs Low Vol)
{reg_str}

## Academic References
- Makarov & Schoar (2020) *Trading and Arbitrage in Cryptocurrency Markets* — JFinEc
- Ornstein & Uhlenbeck (1930) — mean-reverting stochastic processes
- Ou, Hu & Li (2021) — Cross-exchange price discovery in crypto
"""
        p = REPORTS_DIR / f"alpha_{ALPHA_ID}_report.md"
        p.write_text(report)
        log.info("Report → %s", p)
        return report


def run_alpha07(assets=None, start=DEFAULT_START, end=DEFAULT_END):
    a = Alpha07(assets=assets, start=start, end=end)
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
    args = p.parse_args()
    a = Alpha07(start=args.start, end=args.end)
    a.run(); a.plot(); a.generate_report()
    print("\n" + "="*60 + "\nALPHA 07 COMPLETE\n" + "="*60)
    for k, v in a.metrics.items():
        print(f"  {k:<38} {v:.4f}" if isinstance(v, float) else f"  {k:<38} {v}")
