"""
alpha_04_order_flow_imbalance.py
─────────────────────────────────
ALPHA 04 — Order Flow Imbalance Persistence
============================================

HYPOTHESIS
----------
When buy-initiated order flow consistently dominates sell-initiated flow for
3+ consecutive periods, institutional accumulation is occurring.  Persistent
imbalance in one direction predicts next-period price continuation (1–4 periods)
before reverting.

This is grounded in the Glosten-Milgrom (1985) adverse selection framework:
informed traders split their orders over time to minimise price impact.  A
sustained OFI in the same direction is a footprint of informed accumulation
that the market-maker's bid-ask spread cannot immediately absorb.

FORMULA
-------
For each hourly bar:
    OFI_t = (C_t - O_t) / (H_t - L_t + ε)     ∈ [-1, +1]
    (Continuous bulk-volume-classification proxy)

3-period persistence signal:
    Σ₃ = Σ_{k=0}^{2} OFI_{t-k}                 ∈ [-3, +3]

    α₄ = sign(Σ₃) × |Σ₃ / 3|^{0.5}

The square-root dampens sensitivity to extreme/outlier values.
Positive α₄ → expect upward price movement; negative → downward.
Cross-sectional z-score normalisation applied before use.

ASSET CLASS
-----------
Crypto perpetual futures on Binance (hourly bars).
Top-20 crypto by market cap provides a rich cross-section.

REBALANCE FREQUENCY
-------------------
Hourly.  This is a short-horizon microstructure signal.

VALIDATION
----------
• IC at lag 1H, 4H, 12H, 24H — show peak IC and decay
• Hit rate (signal direction matches return) — expect 53–56%
• Signal auto-correlation: persistence > 0 validates accumulation hypothesis
• Sharpe, Max Drawdown (hourly PnL)
• Show sign-symmetry: is the short-side (negative OFI) as strong as the long?

IMPORTANT NOTES
───────────────
Because Binance minute-level data for 20 assets is large, this module uses
hourly candles (1h interval) as the primary granularity.  OHLCV from hourly
bars provides a reasonable proxy for order flow direction via the
(Close-Open)/(High-Low) range-based classifier.

REFERENCES
----------
• Glosten & Milgrom (1985) *Bid, Ask and Transaction Prices in a Specialist
  Market with Heterogeneously Informed Traders* — JFinEc
• Cont, Kukanov & Stoikov (2014) *The Price Impact of Order Book Events* — JFinEc
• Easley & O'Hara (1987) *Price, Trade Size, and Information in Securities Markets*

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
    winsorise,
    information_coefficient,
    information_coefficient_matrix,
    compute_max_drawdown,
    compute_sharpe,
    compute_turnover,
    long_short_portfolio_returns,
    walk_forward_split,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
log = logging.getLogger("Alpha04")

# ── configuration ─────────────────────────────────────────────────────────────
ALPHA_ID          = "04"
ALPHA_NAME        = "OFI_Persistence"
OUTPUT_DIR        = Path("./results")
REPORTS_DIR       = OUTPUT_DIR / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_START     = "2022-01-01"   # 2+ years of hourly data
DEFAULT_END       = "2024-12-31"
OFI_WINDOW        = 3              # persistence window (3 periods)
IC_LAGS_HOURLY    = [1, 2, 3, 4, 6, 12, 24]    # in hours
TOP_PCT           = 0.20
TC_BPS            = 8.0            # per-trade, higher for hourly
IS_FRACTION       = 0.70
INTERVAL          = "1h"           # Binance interval


# ══════════════════════════════════════════════════════════════════════════════
class OFICalculator:
    """
    Computes Order Flow Imbalance (OFI) from OHLCV candles using a continuous
    bulk-volume-classification proxy.

    The range-based formula:
        Z_t = (Close_t - Open_t) / (High_t - Low_t + ε)   ∈ [-1, +1]
        OFI_t = Z_t   (buy_frac = (1 + Z_t)/2, sell_frac = (1 - Z_t)/2)
               so |V_B - V_S| / V = |2×buy_frac - 1| = |Z_t|
               directional: OFI_t = sign(Z_t) × |Z_t|

    This is equivalent to:
        buy_vol  = Volume × (1 + Z_t) / 2
        sell_vol = Volume × (1 - Z_t) / 2
        OFI      = (buy_vol - sell_vol) / Volume = Z_t
    """

    @staticmethod
    def compute_ofi(ohlcv: pd.DataFrame) -> pd.Series:
        """
        Compute per-period OFI ∈ [-1, +1].

        Parameters
        ----------
        ohlcv : DataFrame with columns Open, High, Low, Close, Volume

        Returns
        -------
        pd.Series of OFI values (same index as ohlcv)
        """
        o = ohlcv["Open"]
        h = ohlcv["High"]
        l = ohlcv["Low"]
        c = ohlcv["Close"]

        hl_range = (h - l).replace(0, np.nan)
        z        = (c - o) / hl_range         # ∈ [-1, +1]
        ofi      = z.clip(-1, 1)
        ofi.name = "OFI"
        return ofi

    @staticmethod
    def compute_persistence_signal(ofi: pd.Series, window: int = OFI_WINDOW) -> pd.Series:
        """
        Computes the 3-period persistence signal:
            Σ₃ = sum of OFI over last `window` periods
            α₄ = sign(Σ₃) × |Σ₃ / window|^0.5

        The sign captures direction; the square-root dampens outliers.
        """
        rolling_sum  = ofi.rolling(window, min_periods=window).sum()
        mean_ofi     = rolling_sum / window
        signal       = np.sign(rolling_sum) * np.sqrt(np.abs(mean_ofi))
        signal.name  = "OFI_Persistence"
        return signal


# ══════════════════════════════════════════════════════════════════════════════
class Alpha04:
    """
    Full implementation of Alpha 04 — Order Flow Imbalance Persistence.

    Architecture:
    - Loads hourly OHLCV for crypto universe
    - Computes OFI per asset per hour
    - Computes persistence signal (3-period rolling)
    - Cross-sectional z-score normalisation
    - Hourly IC at multiple lags
    - Hit rate and auto-correlation analysis
    - Signed performance (is short side equally profitable?)
    """

    def __init__(
        self,
        symbols:     List[str] = None,
        start:       str       = DEFAULT_START,
        end:         str       = DEFAULT_END,
        ofi_window:  int       = OFI_WINDOW,
        ic_lags:     List[int] = IC_LAGS_HOURLY,
        top_pct:     float     = TOP_PCT,
        tc_bps:      float     = TC_BPS,
    ):
        self.symbols    = symbols or CRYPTO_UNIVERSE[:10]   # 10 assets for hourly (data-size management)
        self.start      = start
        self.end        = end
        self.ofi_window = ofi_window
        self.ic_lags    = ic_lags
        self.top_pct    = top_pct
        self.tc_bps     = tc_bps

        self._fetcher  = DataFetcher()
        self._ofi_calc = OFICalculator()

        # outputs
        self.ohlcv_dict:  Dict[str, pd.DataFrame] = {}
        self.close:       Optional[pd.DataFrame]  = None
        self.ofi_df:      Optional[pd.DataFrame]  = None
        self.signals:     Optional[pd.DataFrame]  = None
        self.returns:     Optional[pd.DataFrame]  = None
        self.pnl:         Optional[pd.Series]     = None
        self.ic_table:    Optional[pd.DataFrame]  = None
        self.ic_is:       Optional[pd.DataFrame]  = None
        self.ic_oos:      Optional[pd.DataFrame]  = None
        self.hit_rate:    Dict                    = {}
        self.autocorr:    Optional[pd.DataFrame]  = None
        self.sign_analysis: Optional[pd.DataFrame] = None
        self.metrics:     Dict                    = {}

        log.info("Alpha04 initialised | %d symbols | %s→%s | interval=%s",
                 len(self.symbols), start, end, INTERVAL)

    # ── data loading ──────────────────────────────────────────────────────────

    def _load_data(self) -> None:
        log.info("Loading hourly OHLCV for %d symbols …", len(self.symbols))
        for sym in self.symbols:
            try:
                df = self._fetcher.get_crypto_ohlcv(sym, INTERVAL, self.start, self.end)
                if not df.empty:
                    self.ohlcv_dict[sym] = df
                    log.debug("  Loaded %s | %d bars", sym, len(df))
            except Exception as exc:
                log.warning("Failed to load %s: %s", sym, exc)

        if not self.ohlcv_dict:
            raise RuntimeError("No hourly data loaded — check internet connection and Binance API availability.")

        close_frames = {sym: df["Close"] for sym, df in self.ohlcv_dict.items()}
        self.close   = pd.DataFrame(close_frames).sort_index()

        # align to common datetime index
        self.close = self.close.ffill().dropna(how="all", axis=1)
        good = self.close.notna().mean()[self.close.notna().mean() >= 0.80].index
        self.close = self.close[good]

        log.info("Hourly data loaded | %d assets | %d bars", self.close.shape[1], self.close.shape[0])

    # ── OFI computation ───────────────────────────────────────────────────────

    def _compute_ofi(self) -> None:
        """Compute OFI and persistence signal for each symbol."""
        log.info("Computing OFI for %d symbols …", len(self.ohlcv_dict))
        raw_ofi_frames  = {}
        pers_ofi_frames = {}

        for sym, df in self.ohlcv_dict.items():
            if sym not in self.close.columns:
                continue
            df_aligned   = df.reindex(self.close.index).ffill()
            raw_ofi      = OFICalculator.compute_ofi(df_aligned)
            pers_signal  = OFICalculator.compute_persistence_signal(raw_ofi, self.ofi_window)
            raw_ofi_frames[sym]  = raw_ofi
            pers_ofi_frames[sym] = pers_signal

        self.raw_ofi_df = pd.DataFrame(raw_ofi_frames).sort_index()
        self.ofi_df     = pd.DataFrame(pers_ofi_frames).sort_index()
        log.info("OFI computed | shape=%s", self.ofi_df.shape)

    # ── signal construction ───────────────────────────────────────────────────

    def _compute_signal(self) -> None:
        """
        Cross-sectionally z-score normalise the OFI persistence signal.
        Signal ∈ [-∞, +∞] but typically ≈ N(0, 1) after normalisation.
        """
        log.info("Constructing cross-sectional signal …")
        cs_mean  = self.ofi_df.mean(axis=1)
        cs_std   = self.ofi_df.std(axis=1).replace(0, np.nan)
        self.signals = self.ofi_df.subtract(cs_mean, axis=0).divide(cs_std, axis=0)
        self.signals = self.signals.clip(-3, 3)   # cap at ±3σ
        log.info("Signal computed | shape=%s", self.signals.shape)

    # ── hit rate analysis ─────────────────────────────────────────────────────

    def _compute_hit_rate(self) -> None:
        """
        Hit rate: fraction of predictions where signal direction = return direction.
        Computed per lag for long signals, short signals, and combined.
        """
        log.info("Computing hit rates …")
        returns_1h = compute_returns(self.close, 1)   # 1-hour forward return

        results = {}
        for lag in [1, 4, 12, 24]:
            fwd = returns_1h.shift(-lag)
            hits_long   = []
            hits_short  = []

            for date in self.signals.index:
                if date not in fwd.index:
                    continue
                sig_row = self.signals.loc[date].dropna()
                fwd_row = fwd.loc[date].dropna()
                common  = sig_row.index.intersection(fwd_row.index)
                if len(common) < 3:
                    continue

                for asset in common:
                    s = sig_row[asset]
                    r = fwd_row[asset]
                    if s > 0.5:     # strong positive signal
                        hits_long.append(int(r > 0))
                    elif s < -0.5:  # strong negative signal
                        hits_short.append(int(r < 0))

            hit_rate_long  = np.mean(hits_long)  if hits_long  else np.nan
            hit_rate_short = np.mean(hits_short) if hits_short else np.nan
            combined       = np.mean(hits_long + hits_short) if (hits_long or hits_short) else np.nan

            results[f"lag_{lag}h"] = {
                "hit_rate_long":     hit_rate_long,
                "hit_rate_short":    hit_rate_short,
                "hit_rate_combined": combined,
                "n_long":            len(hits_long),
                "n_short":           len(hits_short),
            }

        self.hit_rate = results
        log.info("Hit rate @ lag 1h: combined=%.3f", results.get("lag_1h", {}).get("hit_rate_combined", np.nan))

    # ── auto-correlation analysis ─────────────────────────────────────────────

    def _compute_autocorr(self) -> None:
        """
        Tests the accumulation hypothesis:
        "If OFI is positive (institutional buying), it tends to be positive in
         the NEXT period too — confirming the split-order pattern."
        """
        log.info("Computing OFI auto-correlations …")
        rows = []
        for sym in self.raw_ofi_df.columns:
            ofi = self.raw_ofi_df[sym].dropna()
            ac1  = ofi.autocorr(1)
            ac2  = ofi.autocorr(2)
            ac3  = ofi.autocorr(3)
            ac6  = ofi.autocorr(6)
            ac12 = ofi.autocorr(12)
            rows.append({
                "symbol": sym,
                "AC_lag1": ac1, "AC_lag2": ac2, "AC_lag3": ac3,
                "AC_lag6": ac6, "AC_lag12": ac12,
            })

        self.autocorr = pd.DataFrame(rows).set_index("symbol")
        log.info("Mean AC(1) = %.4f   (>0 confirms accumulation hypothesis)",
                 self.autocorr["AC_lag1"].mean())

    # ── sign asymmetry ────────────────────────────────────────────────────────

    def _compute_sign_analysis(self) -> None:
        """
        Tests whether long side (positive OFI) and short side (negative OFI)
        are equally informative.  Splits into long vs short observations.
        """
        log.info("Computing sign asymmetry …")
        returns_1h = compute_returns(self.close, 1)
        fwd_1h     = returns_1h.shift(-1)

        long_ics, short_ics = [], []
        for date in self.signals.index:
            if date not in fwd_1h.index:
                continue
            sig = self.signals.loc[date].dropna()
            fwd = fwd_1h.loc[date].dropna()
            common = sig.index.intersection(fwd.index)
            if len(common) < 4:
                continue

            long_mask  = sig[common] > 0.5
            short_mask = sig[common] < -0.5

            if long_mask.sum() >= 2:
                ic_long = information_coefficient(sig[common][long_mask], fwd[common][long_mask])
                long_ics.append(ic_long)
            if short_mask.sum() >= 2:
                sig_short = -sig[common][short_mask]   # flip sign
                ic_short = information_coefficient(sig_short, fwd[common][short_mask])
                short_ics.append(ic_short)

        rows = []
        for name, ics in [("Long (positive OFI)", long_ics), ("Short (negative OFI)", short_ics)]:
            arr = np.array([x for x in ics if not np.isnan(x)])
            if len(arr) > 2:
                rows.append({
                    "side": name,
                    "mean_IC": arr.mean(),
                    "std_IC":  arr.std(ddof=1),
                    "n_obs":   len(arr),
                    "t_stat":  arr.mean() / (arr.std(ddof=1) / np.sqrt(len(arr))),
                })
            else:
                rows.append({"side": name, "mean_IC": np.nan, "std_IC": np.nan, "n_obs": 0, "t_stat": np.nan})

        self.sign_analysis = pd.DataFrame(rows).set_index("side")

    # ── main pipeline ─────────────────────────────────────────────────────────

    def run(self) -> "Alpha04":
        self._load_data()
        self._compute_ofi()
        self._compute_signal()

        self.returns = compute_returns(self.close, 1)   # hourly returns
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

        # Additional analyses
        self._compute_hit_rate()
        self._compute_autocorr()
        self._compute_sign_analysis()

        # Portfolio PnL (hourly)
        log.info("Computing hourly PnL …")
        self.pnl = long_short_portfolio_returns(
            self.signals, self.returns, top_pct=self.top_pct, transaction_cost_bps=self.tc_bps
        )

        self._compute_metrics()
        return self

    # ── metrics ───────────────────────────────────────────────────────────────

    def _compute_metrics(self) -> None:
        pnl  = self.pnl.dropna()
        ic1h_is  = self.ic_is.loc[1, "mean_IC"]  if 1 in self.ic_is.index  else np.nan
        ic1h_oos = self.ic_oos.loc[1, "mean_IC"] if 1 in self.ic_oos.index else np.nan
        hr_1h    = self.hit_rate.get("lag_1h", {}).get("hit_rate_combined", np.nan)
        mean_ac1 = self.autocorr["AC_lag1"].mean() if self.autocorr is not None else np.nan

        self.metrics = {
            "alpha_id":           ALPHA_ID,
            "alpha_name":         ALPHA_NAME,
            "universe":           "Crypto Hourly",
            "n_assets":           self.close.shape[1],
            "n_hours":            self.close.shape[0],
            "IC_mean_IS_lag1h":   float(ic1h_is),
            "IC_mean_OOS_lag1h":  float(ic1h_oos),
            "ICIR_IS_1h":         float(self.ic_is.loc[1,  "ICIR"]) if 1 in self.ic_is.index  else np.nan,
            "ICIR_OOS_1h":        float(self.ic_oos.loc[1, "ICIR"]) if 1 in self.ic_oos.index else np.nan,
            "Sharpe_hourly":      compute_sharpe(pnl, periods_per_year=365*24),
            "MaxDrawdown":        compute_max_drawdown(pnl),
            "HitRate_1h":         float(hr_1h),
            "Mean_OFI_AutoCorr1": float(mean_ac1),
        }

        log.info("─── Alpha 04 Metrics ───────────────────────────────────")
        for k, v in self.metrics.items():
            log.info("  %-35s = %s", k, f"{v:.5f}" if isinstance(v, float) else v)

    # ── visualisation ─────────────────────────────────────────────────────────

    def plot(self, save: bool = True) -> None:
        """
        5-panel figure:
        1. IC decay curve (hourly lags)
        2. OFI time series for BTC
        3. Auto-correlation bar chart
        4. Hit rate by lag
        5. Sign asymmetry (long vs short)
        """
        fig = plt.figure(figsize=(20, 16))
        gs  = gridspec.GridSpec(3, 2, hspace=0.45, wspace=0.30)

        # ── Panel 1: IC Decay ─────────────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        lags_plot = [l for l in self.ic_lags if l in self.ic_table.index]
        ic_is_vals  = [self.ic_is.loc[l,  "mean_IC"] if l in self.ic_is.index  else np.nan for l in lags_plot]
        ic_oos_vals = [self.ic_oos.loc[l, "mean_IC"] if l in self.ic_oos.index else np.nan for l in lags_plot]
        ax1.plot(lags_plot, ic_is_vals,  marker="o", label="In-Sample",     color="#2ca02c", linewidth=2)
        ax1.plot(lags_plot, ic_oos_vals, marker="s", label="Out-of-Sample", color="#d62728", linewidth=2, linestyle="--")
        ax1.axhline(0, color="black", linewidth=0.7)
        ax1.set_xlabel("Lag (hours)")
        ax1.set_ylabel("Mean IC")
        ax1.set_title("Alpha 04 — IC Decay Curve (Hourly)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # ── Panel 2: OFI Time Series ───────────────────────────────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        btc_syms = [s for s in self.raw_ofi_df.columns if "BTC" in s]
        if btc_syms:
            sym    = btc_syms[0]
            ofi_ts = self.raw_ofi_df[sym].dropna()
            price  = self.close[sym].dropna()
            # show last 500 hours for clarity
            ofi_ts = ofi_ts.tail(500)
            price  = price.tail(500)
            ax2_r  = ax2.twinx()
            ax2.bar(ofi_ts.index, ofi_ts.values, alpha=0.6, width=0.04,
                    color=["green" if v >= 0 else "red" for v in ofi_ts.values], label="OFI")
            ax2.axhline(0, color="black", linewidth=0.6)
            ax2_r.plot(price.index, price.values, color="navy", linewidth=1.0, alpha=0.7, label="Price")
            ax2.set_ylabel("OFI (Buy - Sell)")
            ax2_r.set_ylabel("Price (USDT)")
            ax2.set_title(f"Alpha 04 — OFI vs Price ({sym})\nLast 500 Hourly Bars")
            ax2.legend(loc="upper left")

        # ── Panel 3: Auto-correlation ──────────────────────────────────────────
        ax3 = fig.add_subplot(gs[1, 0])
        if self.autocorr is not None:
            ac_lags  = ["AC_lag1","AC_lag2","AC_lag3","AC_lag6","AC_lag12"]
            ac_means = [self.autocorr[c].mean() for c in ac_lags]
            ac_stds  = [self.autocorr[c].std() / np.sqrt(len(self.autocorr)) for c in ac_lags]
            lag_labels = ["1h","2h","3h","6h","12h"]
            ax3.bar(lag_labels, ac_means, yerr=ac_stds, capsize=5, color="#8c564b", alpha=0.8, edgecolor="black")
            ax3.axhline(0, color="black", linewidth=0.8)
            ax3.axhline(2/np.sqrt(len(self.close)), color="red", linewidth=1.0, linestyle="--",
                        alpha=0.7, label="5% significance")
            ax3.axhline(-2/np.sqrt(len(self.close)), color="red", linewidth=1.0, linestyle="--", alpha=0.7)
            ax3.set_ylabel("Mean Auto-Correlation")
            ax3.set_title("OFI Auto-Correlation by Lag\n(>0 confirms accumulation hypothesis)")
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis="y")

        # ── Panel 4: Hit Rate by Lag ───────────────────────────────────────────
        ax4 = fig.add_subplot(gs[1, 1])
        hr_lags = list(self.hit_rate.keys())
        hr_long  = [self.hit_rate[k]["hit_rate_long"]     for k in hr_lags]
        hr_short = [self.hit_rate[k]["hit_rate_short"]    for k in hr_lags]
        hr_comb  = [self.hit_rate[k]["hit_rate_combined"] for k in hr_lags]
        x = np.arange(len(hr_lags))
        w = 0.25
        ax4.bar(x - w,   hr_long,  w, label="Long  (pos OFI)", color="#1f77b4", alpha=0.8)
        ax4.bar(x,       hr_short, w, label="Short (neg OFI)", color="#ff7f0e", alpha=0.8)
        ax4.bar(x + w,   hr_comb,  w, label="Combined",        color="#2ca02c", alpha=0.8)
        ax4.axhline(0.5, color="black", linewidth=1.2, linestyle="--", label="Random (50%)")
        ax4.set_xticks(x)
        ax4.set_xticklabels([k.replace("lag_", "").upper() for k in hr_lags])
        ax4.set_ylim(0.3, 0.75)
        ax4.set_ylabel("Hit Rate")
        ax4.set_title("Alpha 04 — Hit Rate by Lag\n(Target: 53–56% above random)")
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3, axis="y")

        # ── Panel 5: Cumulative PnL ────────────────────────────────────────────
        ax5 = fig.add_subplot(gs[2, :])
        cum_pnl = self.pnl.dropna().cumsum()
        ax5.plot(cum_pnl.index, cum_pnl.values, linewidth=1.5, color="#1f77b4", label="OFI Persistence L/S")
        roll_max = cum_pnl.cummax()
        dd = cum_pnl - roll_max
        ax5.fill_between(dd.index, dd.values, 0, where=dd.values < 0, alpha=0.3, color="red", label="Drawdown")
        ax5.axhline(0, color="black", linewidth=0.6)
        ax5.set_title("Alpha 04 — Cumulative PnL (Hourly, Net of 8 bps TC)")
        ax5.set_ylabel("Cumulative Return")
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        plt.suptitle(
            f"ALPHA 04 — Order Flow Imbalance Persistence\n"
            f"Sharpe={self.metrics.get('Sharpe_hourly', np.nan):.2f}  "
            f"IC(OOS,1h)={self.metrics.get('IC_mean_OOS_lag1h', np.nan):.4f}  "
            f"HitRate(1h)={self.metrics.get('HitRate_1h', np.nan):.3f}  "
            f"OFI_AutoCorr(1h)={self.metrics.get('Mean_OFI_AutoCorr1', np.nan):.4f}",
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
        autocorr_str = self.autocorr.to_markdown(floatfmt=".4f") if self.autocorr is not None else "N/A"
        sign_str     = self.sign_analysis.to_markdown(floatfmt=".4f") if self.sign_analysis is not None else "N/A"

        # format hit rate table
        hr_rows = []
        for lag, stats in self.hit_rate.items():
            hr_rows.append({"lag": lag, **stats})
        hr_str = pd.DataFrame(hr_rows).set_index("lag").to_markdown(floatfmt=".4f") if hr_rows else "N/A"

        report = f"""# Alpha {ALPHA_ID}: {ALPHA_NAME.replace('_', ' ')}

## Hypothesis
Persistent order flow imbalance over 3 consecutive hourly bars signals
institutional accumulation (split order pattern).  The theoretical basis is
the Glosten-Milgrom adverse selection model: informed traders split large
orders to minimise price impact, creating sustained directional OFI.
This predicts price continuation for 1–4 hours before reverting.

## Expression (Python)
```python
# OFI computation (range-based bulk classification)
price_range = (high - low).replace(0, np.nan)
Z_t         = (close - open) / price_range              # ∈ [-1, +1]
OFI_t       = Z_t.clip(-1, 1)                           # directional flow

# 3-period persistence signal
rolling_sum  = OFI_t.rolling(3, min_periods=3).sum()    # ∈ [-3, +3]
mean_ofi     = rolling_sum / 3
alpha_04_raw = sign(rolling_sum) * sqrt(abs(mean_ofi))  # dampened

# cross-sectional z-score
cs_mean  = alpha_04_raw.mean(axis=1)
cs_std   = alpha_04_raw.std(axis=1)
alpha_04 = (alpha_04_raw - cs_mean) / cs_std            # ≈ N(0,1)
```

## Parameters
| Parameter         | Value                   |
|-------------------|-------------------------|
| OFI window        | {self.ofi_window} periods (hours) |
| Rebalance         | Hourly                  |
| Universe          | Crypto top-{len(self.symbols)} (Binance hourly) |
| Long/Short pct    | {self.top_pct*100:.0f}% |
| TC assumption     | {self.tc_bps:.0f} bps (round-trip) |

## Performance Summary

| Metric                    | Value                  |
|---------------------------|------------------------|
| Sharpe (annualised, hourly)| {self.metrics.get('Sharpe_hourly', np.nan):.3f} |
| Max Drawdown              | {self.metrics.get('MaxDrawdown', np.nan)*100:.2f}% |
| IC (IS)  @ 1h             | {self.metrics.get('IC_mean_IS_lag1h', np.nan):.5f} |
| IC (OOS) @ 1h             | {self.metrics.get('IC_mean_OOS_lag1h', np.nan):.5f} |
| ICIR (IS)  @ 1h           | {self.metrics.get('ICIR_IS_1h', np.nan):.3f} |
| ICIR (OOS) @ 1h           | {self.metrics.get('ICIR_OOS_1h', np.nan):.3f} |
| Hit Rate @ 1h             | {self.metrics.get('HitRate_1h', np.nan):.3f} |
| Mean OFI Auto-Corr(1h)    | {self.metrics.get('Mean_OFI_AutoCorr1', np.nan):.4f} |

## IC Decay Table (Full Sample)
{ic_table_str}

## In-Sample IC by Lag
{ic_is_str}

## Out-of-Sample IC by Lag
{ic_oos_str}

## Hit Rate by Lag
{hr_str}

## OFI Auto-Correlation (Accumulation Hypothesis Test)
{autocorr_str}

*Auto-corr > 0 validates that OFI is positively serially correlated — consistent with split-order
institutional accumulation as described by Glosten-Milgrom.*

## Sign Asymmetry (Long vs Short Side)
{sign_str}

## Transaction Cost Break-Even
At hourly rebalancing and {self.tc_bps:.0f} bps TC, break-even requires mean gross return
per trade > {self.tc_bps / 10_000 * 100:.3f}% per hour.

## Academic References
- Glosten & Milgrom (1985) *Bid, Ask and Transaction Prices in a Specialist Market* — JFinEc
- Cont, Kukanov & Stoikov (2014) *The Price Impact of Order Book Events* — JFinEc
- Easley & O'Hara (1987) *Price, Trade Size, and Information in Securities Markets* — JFinEc
"""
        report_path = REPORTS_DIR / f"alpha_{ALPHA_ID}_report.md"
        report_path.write_text(report)
        log.info("Report saved → %s", report_path)
        return report


# ══════════════════════════════════════════════════════════════════════════════

def run_alpha04(
    symbols: List[str] = None,
    start:   str       = DEFAULT_START,
    end:     str       = DEFAULT_END,
) -> Alpha04:
    alpha = Alpha04(symbols=symbols, start=start, end=end)
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
    parser = argparse.ArgumentParser(description="Alpha 04 — OFI Persistence")
    parser.add_argument("--start",   default=DEFAULT_START)
    parser.add_argument("--end",     default=DEFAULT_END)
    parser.add_argument("--symbols", nargs="+", default=None)
    args = parser.parse_args()

    alpha = Alpha04(symbols=args.symbols, start=args.start, end=args.end)
    alpha.run()
    alpha.plot()
    alpha.generate_report()

    print("\n" + "=" * 60)
    print("ALPHA 04 COMPLETE")
    print("=" * 60)
    for k, v in alpha.metrics.items():
        print(f"  {k:<40} {v:.5f}" if isinstance(v, float) else f"  {k:<40} {v}")
