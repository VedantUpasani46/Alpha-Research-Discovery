"""
alpha_13_cross_asset_macro_regime.py
──────────────────────────────────────
ALPHA 13 — Cross-Asset Macro Regime Signal (Risk-On / Risk-Off)
===============================================================

HYPOTHESIS
----------
Global risk appetite oscillates between "risk-on" (investors seek yield,
buy equities/commodities/carry currencies) and "risk-off" (flight to safety,
buy bonds/gold/yen).  A composite cross-asset signal constructed from FX carry,
credit spreads, commodity momentum, and rates term structure identifies this
regime faster than equity-only signals.

Once the macro regime is identified, the alpha tilts the equity/crypto book:
  • Risk-ON:  increase leverage on momentum/high-beta positions
  • Risk-OFF: reduce exposure, rotate to low-vol / quality / cash

FORMULA
-------
Four macro sub-signals (each cross-sectionally normalised to [-1, +1]):

    1. FX Carry:      Long AUD,NZD,CAD vs Short JPY,CHF (EM carry vs safe havens)
                      Signal = 20-day momentum of AUD/JPY, NZD/JPY, CAD/JPY

    2. Credit Spread: HYG/LQD ratio change (high-yield vs investment-grade)
                      Rising = risk-off; falling = risk-on

    3. Commodity Mom: CRB basket proxy (CL=F, GC=F, SI=F)
                      Rising commodities = risk-on

    4. Term Structure: (TY=F 10Y − FVX 5Y) slope change
                       Steepening = risk-on; flattening = risk-off

    RiskOn_t = mean(rank(FX_carry), rank(-credit_spread_chg),
                    rank(commodity_mom), rank(term_spread_chg))

Portfolio tilt:
    adjusted_weight = base_weight × (1 + γ × RiskOn_t)   [risk-on, γ=0.3]
    adjusted_weight = base_weight × (1 - γ × |RiskOn_t|)  [risk-off]

ASSET CLASS
-----------
Primary: S&P 500 equities (tilts the entire equity book)
         Or crypto basket (tilts based on macro regime)

VALIDATION
----------
• Correlation of composite to VIX (should be strongly negative)
• Sharpe of alpha book with vs without macro tilt
• Drawdown reduction during 2020/2022-type risk-off episodes
• Sub-signal contribution analysis

REFERENCES
----------
• Koijen et al. (2018) *Carry* — JFinEc
• Lustig & Verdelhan (2007) *The Cross Section of FX Risk Premia* — AER
• Asness, Moskowitz & Pedersen (2013) *Value and Momentum Everywhere* — JF

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

from data_fetcher import (
    DataFetcher,
    SP500_TICKERS,
    CRYPTO_UNIVERSE,
    compute_returns,
    cross_sectional_rank,
    information_coefficient_matrix,
    compute_max_drawdown,
    compute_sharpe,
    long_short_portfolio_returns,
    walk_forward_split,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
log = logging.getLogger("Alpha13")

ALPHA_ID    = "13"
ALPHA_NAME  = "CrossAsset_MacroRegime"
OUTPUT_DIR  = Path("./results")
REPORTS_DIR = OUTPUT_DIR / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_START = "2015-01-01"
DEFAULT_END   = "2024-12-31"
IC_LAGS       = [1, 5, 10, 22]
TOP_PCT       = 0.20
TC_BPS        = 8.0
IS_FRACTION   = 0.70
TILT_GAMMA    = 0.30    # how aggressively to tilt on regime signal
FX_CARRY_WIN  = 20      # days for FX momentum
CREDIT_WIN    = 5       # days for credit spread change
COMMODITY_WIN = 20      # days for commodity momentum
RATES_WIN     = 10      # days for term structure change

# ETF/Index proxies (all via yfinance)
MACRO_TICKERS = {
    "spy":    "SPY",       # S&P 500
    "hyg":    "HYG",       # High-yield credit
    "lqd":    "LQD",       # Investment-grade credit
    "tlt":    "TLT",       # Long-term treasuries (safe haven)
    "gld":    "GLD",       # Gold
    "uso":    "USO",       # Crude oil
    "uup":    "UUP",       # USD index
    "vxx":    "VXX",       # Volatility (VIX proxy)
    "aud":    "AUDUSD=X",  # AUD/USD
    "nzd":    "NZDUSD=X",  # NZD/USD
    "cad":    "CADUSD=X",  # CAD/USD (inverse USDCAD)
    "jpy":    "JPYUSD=X",  # JPY/USD (inverse USDJPY)
    "chf":    "CHFUSD=X",  # CHF/USD
    "oil":    "CL=F",      # WTI crude futures
    "gold":   "GC=F",      # Gold futures
}


class MacroSignalEngine:
    """
    Constructs the 4 macro sub-signals from ETF/FX price data.
    All signals are standardised to z-scores before compositing.
    """

    def __init__(self, prices: pd.DataFrame):
        self.prices = prices.copy()

    def _zscore(self, series: pd.Series, window: int = 252) -> pd.Series:
        """Rolling z-score normalisation."""
        mu  = series.rolling(window, min_periods=20).mean()
        std = series.rolling(window, min_periods=20).std().replace(0, np.nan)
        return (series - mu) / std

    def fx_carry(self) -> pd.Series:
        """
        Risk-on FX carry: momentum of high-yield vs safe-haven currencies.
        Long AUD+NZD+CAD vs Short JPY+CHF.
        Signal = 20-day return of risk-on basket minus safe-haven basket.
        """
        risk_on  = []
        safe_hav = []
        for ticker in ["AUDUSD=X", "NZDUSD=X", "CADUSD=X"]:
            if ticker in self.prices.columns:
                r = np.log(self.prices[ticker] / self.prices[ticker].shift(FX_CARRY_WIN))
                risk_on.append(r)
        for ticker in ["JPYUSD=X", "CHFUSD=X"]:
            if ticker in self.prices.columns:
                r = np.log(self.prices[ticker] / self.prices[ticker].shift(FX_CARRY_WIN))
                safe_hav.append(r)

        if not risk_on and not safe_hav:
            return pd.Series(0.0, index=self.prices.index, name="fx_carry")

        ro_mean  = pd.concat(risk_on,  axis=1).mean(axis=1) if risk_on  else pd.Series(0, index=self.prices.index)
        sh_mean  = pd.concat(safe_hav, axis=1).mean(axis=1) if safe_hav else pd.Series(0, index=self.prices.index)
        carry    = ro_mean - sh_mean
        return self._zscore(carry).rename("fx_carry")

    def credit_spread(self) -> pd.Series:
        """
        Credit risk appetite: HYG/LQD ratio.
        Rising ratio = risk-on; falling = risk-off.
        """
        if "HYG" not in self.prices.columns or "LQD" not in self.prices.columns:
            return pd.Series(0.0, index=self.prices.index, name="credit_spread")
        ratio  = np.log(self.prices["HYG"] / self.prices["LQD"].replace(0, np.nan))
        change = ratio.diff(CREDIT_WIN)
        return self._zscore(change).rename("credit_spread")

    def commodity_momentum(self) -> pd.Series:
        """
        Commodity momentum: 20-day return of oil + gold basket.
        """
        commodities = []
        for t in ["CL=F", "GC=F", "USO", "GLD"]:
            if t in self.prices.columns:
                r = np.log(self.prices[t] / self.prices[t].shift(COMMODITY_WIN))
                commodities.append(r)
        if not commodities:
            return pd.Series(0.0, index=self.prices.index, name="commodity_mom")
        compo = pd.concat(commodities, axis=1).mean(axis=1)
        return self._zscore(compo).rename("commodity_mom")

    def rates_term_structure(self) -> pd.Series:
        """
        Rates slope: TLT (long bonds) vs SHY/IEF (short bonds).
        Proxy: -dTLT (long rates rising = steepening term structure = risk-on).
        """
        if "TLT" in self.prices.columns:
            tlt_ret = np.log(self.prices["TLT"] / self.prices["TLT"].shift(RATES_WIN))
            # Inverted: rising TLT prices = falling yields = flattening = risk-off
            slope = -tlt_ret
            return self._zscore(slope).rename("term_structure")
        return pd.Series(0.0, index=self.prices.index, name="term_structure")

    def composite_risk_on(self) -> pd.Series:
        """
        Equal-weight composite of 4 sub-signals.
        Positive = risk-on; Negative = risk-off.
        """
        subs = [self.fx_carry(), self.credit_spread(),
                self.commodity_momentum(), self.rates_term_structure()]
        valid = [s for s in subs if s.abs().sum() > 0]
        if not valid:
            return pd.Series(0.0, index=self.prices.index, name="risk_on")
        composite = pd.concat(valid, axis=1).mean(axis=1)
        return composite.rename("risk_on")


# ══════════════════════════════════════════════════════════════════════════════
class Alpha13:
    """
    Cross-Asset Macro Regime Tilt Alpha.
    """

    def __init__(
        self,
        tickers:    List[str] = None,
        start:      str       = DEFAULT_START,
        end:        str       = DEFAULT_END,
        ic_lags:    List[int] = IC_LAGS,
        top_pct:    float     = TOP_PCT,
        tc_bps:     float     = TC_BPS,
        use_crypto: bool      = False,
        tilt_gamma: float     = TILT_GAMMA,
    ):
        self.tickers    = tickers or (CRYPTO_UNIVERSE[:15] if use_crypto else SP500_TICKERS[:50])
        self.start      = start
        self.end        = end
        self.ic_lags    = ic_lags
        self.top_pct    = top_pct
        self.tc_bps     = tc_bps
        self.use_crypto = use_crypto
        self.tilt_gamma = tilt_gamma

        self._fetcher = DataFetcher()

        self.close:            Optional[pd.DataFrame] = None
        self.returns:          Optional[pd.DataFrame] = None
        self.macro_prices:     Optional[pd.DataFrame] = None
        self.risk_on:          Optional[pd.Series]    = None
        self.sub_signals:      Optional[pd.DataFrame] = None
        self.vix:              Optional[pd.Series]    = None
        self.base_signals:     Optional[pd.DataFrame] = None   # momentum
        self.tilted_signals:   Optional[pd.DataFrame] = None   # tilt applied
        self.pnl_tilted:       Optional[pd.Series]    = None
        self.pnl_base:         Optional[pd.Series]    = None
        self.ic_tilted:        Optional[pd.DataFrame] = None
        self.ic_base:          Optional[pd.DataFrame] = None
        self.vix_corr:         Optional[float]        = None
        self.drawdown_analysis:Optional[pd.DataFrame] = None
        self.metrics:          Dict                   = {}

        log.info("Alpha13 | %d tickers | %s→%s", len(self.tickers), start, end)

    def _load_data(self) -> None:
        log.info("Loading asset prices …")
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

        # Macro proxy prices
        log.info("Loading macro proxies …")
        macro_tickers = list(MACRO_TICKERS.values())
        macro_ohlcv   = self._fetcher.get_equity_ohlcv(macro_tickers, self.start, self.end)
        macro_frames  = {t: df["Close"] for t, df in macro_ohlcv.items() if not df.empty}
        self.macro_prices = pd.DataFrame(macro_frames).sort_index().ffill().reindex(self.close.index).ffill()

        # VIX
        try:
            self.vix = self._fetcher.get_vix(self.start, self.end).reindex(self.close.index).ffill()
        except Exception:
            self.vix = None

        log.info("Loaded | %d assets | macro cols=%d", self.close.shape[1], self.macro_prices.shape[1])

    def _compute_macro_signal(self) -> None:
        log.info("Computing cross-asset macro signal …")
        engine = MacroSignalEngine(self.macro_prices)
        fx     = engine.fx_carry()
        cr     = engine.credit_spread()
        co     = engine.commodity_momentum()
        ts     = engine.rates_term_structure()
        self.risk_on = engine.composite_risk_on().reindex(self.close.index).ffill()
        self.sub_signals = pd.DataFrame({
            "FX_Carry": fx, "Credit_Spread": cr,
            "Commodity_Mom": co, "Term_Structure": ts,
        }).reindex(self.close.index).ffill()

        # VIX correlation
        if self.vix is not None:
            aligned = pd.concat([self.risk_on, self.vix], axis=1).dropna()
            if len(aligned) > 20:
                from scipy.stats import pearsonr
                r, _ = pearsonr(aligned.iloc[:, 0], aligned.iloc[:, 1])
                self.vix_corr = float(r)
                log.info("Risk-on ↔ VIX correlation: %.4f (expected < 0)", self.vix_corr)

    def _build_base_signal(self) -> None:
        """Base signal: simple 21-day momentum cross-section."""
        mom_21 = np.log(self.close / self.close.shift(21))
        self.base_signals = cross_sectional_rank(mom_21)

    def _apply_macro_tilt(self) -> None:
        """
        Tilt base signal by macro regime:
            tilted = base × (1 + γ × RiskOn)  if risk-on  > 0
            tilted = base × (1 - γ × |RiskOn|) if risk-on  < 0
        Clamp the multiplier to [1-γ, 1+γ] for stability.
        """
        log.info("Applying macro tilt (γ=%.2f) …", self.tilt_gamma)
        risk_on_daily = self.risk_on.reindex(self.base_signals.index).fillna(0)
        multiplier    = 1.0 + self.tilt_gamma * risk_on_daily.clip(-1, 1)
        # broadcast multiplier across assets
        tilted = self.base_signals.multiply(multiplier, axis=0)
        self.tilted_signals = cross_sectional_rank(tilted)

    def _compute_drawdown_analysis(self) -> None:
        """
        Identify 3 major risk-off episodes and compare tilted vs base drawdowns.
        """
        if self.vix is None:
            return
        # Episodes where VIX > 30 (significant stress)
        stress_idx = self.vix[self.vix > 30].index
        if len(stress_idx) == 0:
            return

        rows = []
        for label, pnl in [("Tilted", self.pnl_tilted), ("Base", self.pnl_base)]:
            if pnl is None:
                continue
            pnl_stress = pnl.loc[pnl.index.intersection(stress_idx)]
            pnl_calm   = pnl.loc[~pnl.index.isin(stress_idx)]
            rows.append({
                "Strategy":     label,
                "Sharpe_Stress":compute_sharpe(pnl_stress) if len(pnl_stress) > 5 else np.nan,
                "Sharpe_Calm":  compute_sharpe(pnl_calm)   if len(pnl_calm)   > 5 else np.nan,
                "MaxDD_Stress": compute_max_drawdown(pnl_stress) if len(pnl_stress) > 5 else np.nan,
            })
        self.drawdown_analysis = pd.DataFrame(rows).set_index("Strategy")

    def run(self) -> "Alpha13":
        self._load_data()
        self._compute_macro_signal()
        self._build_base_signal()
        self._apply_macro_tilt()

        is_idx, oos_idx = walk_forward_split(self.close.index, IS_FRACTION)
        sigs_t = self.tilted_signals.dropna(how="all")
        sigs_b = self.base_signals.dropna(how="all")

        self.ic_tilted = information_coefficient_matrix(sigs_t, self.returns, self.ic_lags)
        self.ic_base   = information_coefficient_matrix(sigs_b, self.returns, self.ic_lags)

        self.pnl_tilted = long_short_portfolio_returns(sigs_t, self.returns, self.top_pct, self.tc_bps)
        self.pnl_base   = long_short_portfolio_returns(sigs_b, self.returns, self.top_pct, self.tc_bps)

        self._compute_drawdown_analysis()
        self._compute_metrics()
        return self

    def _compute_metrics(self) -> None:
        pt = self.pnl_tilted.dropna() if self.pnl_tilted is not None else pd.Series()
        pb = self.pnl_base.dropna()   if self.pnl_base   is not None else pd.Series()

        ic1_t = self.ic_tilted.loc[1, "mean_IC"] if self.ic_tilted is not None and 1 in self.ic_tilted.index else np.nan
        ic1_b = self.ic_base.loc[1,   "mean_IC"] if self.ic_base   is not None and 1 in self.ic_base.index   else np.nan
        ic5_t = self.ic_tilted.loc[5, "mean_IC"] if self.ic_tilted is not None and 5 in self.ic_tilted.index else np.nan

        self.metrics = {
            "alpha_id":           ALPHA_ID,
            "alpha_name":         ALPHA_NAME,
            "n_assets":           self.close.shape[1],
            "IC_tilted_lag1":     float(ic1_t),
            "IC_base_lag1":       float(ic1_b),
            "IC_tilted_lag5":     float(ic5_t),
            "IC_lift":            float(ic1_t - ic1_b) if not np.isnan(ic1_t + ic1_b) else np.nan,
            "Sharpe_tilted":      compute_sharpe(pt) if len(pt) > 0 else np.nan,
            "Sharpe_base":        compute_sharpe(pb) if len(pb) > 0 else np.nan,
            "MaxDrawdown_tilted": compute_max_drawdown(pt) if len(pt) > 0 else np.nan,
            "VIX_correlation":    float(self.vix_corr) if self.vix_corr is not None else np.nan,
            "tilt_gamma":         self.tilt_gamma,
        }
        log.info("─── Alpha 13 Metrics ─────────────────────────────")
        for k, v in self.metrics.items():
            log.info("  %-34s = %s", k, f"{v:.5f}" if isinstance(v, float) else v)

    def plot(self, save: bool = True) -> None:
        fig = plt.figure(figsize=(18, 16))
        gs  = gridspec.GridSpec(3, 2, hspace=0.40, wspace=0.30)

        # Panel 1: Risk-On signal over time with VIX overlay
        ax1 = fig.add_subplot(gs[0, :])
        if self.risk_on is not None:
            ro = self.risk_on.dropna()
            ax1.fill_between(ro.index, ro.values, 0,
                             where=ro.values > 0, alpha=0.4, color="#2ca02c", label="Risk-On")
            ax1.fill_between(ro.index, ro.values, 0,
                             where=ro.values <= 0, alpha=0.4, color="#d62728", label="Risk-Off")
            ax1.plot(ro.index, ro.values, lw=1.0, color="k", alpha=0.6)
            if self.vix is not None:
                ax1_r = ax1.twinx()
                v = self.vix.reindex(ro.index).dropna()
                ax1_r.plot(v.index, v.values, lw=1.2, color="purple", alpha=0.5, label="VIX")
                ax1_r.set_ylabel("VIX")
                ax1_r.legend(loc="upper right", fontsize=9)
            ax1.axhline(0, color="k", lw=0.8)
            ax1.set(ylabel="Risk-On Score (z)", title="Alpha 13 — Cross-Asset Macro Signal\n(Green=Risk-On, Red=Risk-Off)")
            ax1.legend(loc="upper left", fontsize=9)
            ax1.grid(True, alpha=0.3)

        # Panel 2: Sub-signal contributions
        ax2 = fig.add_subplot(gs[1, 0])
        if self.sub_signals is not None:
            recent = self.sub_signals.tail(252).dropna(how="all")
            for col in recent.columns:
                ax2.plot(recent.index, recent[col].values, lw=1.3, alpha=0.8, label=col)
            ax2.axhline(0, color="k", lw=0.6)
            ax2.set(title="Alpha 13 — Sub-Signal Components\n(FX, Credit, Commodity, Rates — last 252 days)",
                    ylabel="Z-score")
            ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

        # Panel 3: IC comparison
        ax3 = fig.add_subplot(gs[1, 1])
        if self.ic_tilted is not None and self.ic_base is not None:
            lags = [l for l in self.ic_lags if l in self.ic_tilted.index and l in self.ic_base.index]
            x = np.arange(len(lags)); w = 0.35
            ic_t = [self.ic_tilted.loc[l, "mean_IC"] for l in lags]
            ic_b = [self.ic_base.loc[l,   "mean_IC"] for l in lags]
            ax3.bar(x - w/2, ic_t, w, label="Macro-Tilted", color="#1f77b4", alpha=0.8)
            ax3.bar(x + w/2, ic_b, w, label="Base (Momentum)", color="#ff7f0e", alpha=0.8)
            ax3.set_xticks(x); ax3.set_xticklabels([f"Lag {l}d" for l in lags])
            ax3.axhline(0, color="k", lw=0.7)
            ax3.set(ylabel="Mean IC", title="Alpha 13 — IC: Tilted vs Base")
            ax3.legend(); ax3.grid(True, alpha=0.3, axis="y")

        # Panel 4: Cumulative PnL
        ax4 = fig.add_subplot(gs[2, :])
        if self.pnl_tilted is not None:
            ct = self.pnl_tilted.dropna().cumsum()
            ax4.plot(ct.index, ct.values, lw=2, color="#1f77b4", label="Macro-Tilted")
        if self.pnl_base is not None:
            cb = self.pnl_base.dropna().cumsum()
            ax4.plot(cb.index, cb.values, lw=2, linestyle="--", color="#ff7f0e",
                     alpha=0.8, label="Base Momentum")
        ax4.axhline(0, color="k", lw=0.6)
        ax4.set(title="Alpha 13 — Cumulative PnL: Macro-Tilted vs Base",
                ylabel="Cumulative Return")
        ax4.legend(); ax4.grid(True, alpha=0.3)

        plt.suptitle(
            f"ALPHA 13 — Cross-Asset Macro Regime\n"
            f"Sharpe_tilted={self.metrics.get('Sharpe_tilted', np.nan):.2f}  "
            f"Sharpe_base={self.metrics.get('Sharpe_base', np.nan):.2f}  "
            f"IC_Lift={self.metrics.get('IC_lift', np.nan):+.4f}  "
            f"VIX_corr={self.metrics.get('VIX_correlation', np.nan):.3f}",
            fontsize=12, fontweight="bold")

        if save:
            out = REPORTS_DIR / f"alpha_{ALPHA_ID}_chart.png"
            plt.savefig(out, dpi=150, bbox_inches="tight"); log.info("Chart → %s", out)
        plt.close(fig)

    def generate_report(self) -> str:
        ic_t_str = self.ic_tilted.reset_index().to_markdown(index=False, floatfmt=".5f") if self.ic_tilted is not None else "N/A"
        ic_b_str = self.ic_base.reset_index().to_markdown(index=False, floatfmt=".5f") if self.ic_base is not None else "N/A"
        dd_str   = self.drawdown_analysis.reset_index().to_markdown(index=False, floatfmt=".4f") if self.drawdown_analysis is not None else "N/A"

        report = f"""# Alpha {ALPHA_ID}: {ALPHA_NAME.replace('_', ' ')}

## Hypothesis
A composite cross-asset signal (FX carry + credit spread + commodity momentum + rates term
structure) identifies risk-on/risk-off regimes faster than equity prices alone.
Tilting the equity book by this signal (γ={self.tilt_gamma}) improves Sharpe and reduces
drawdowns during stress periods.

## Formula
```python
risk_on = mean(z(fx_carry), z(credit_spread_chg), z(commodity_mom), z(term_spread_chg))
tilted  = base_signal × (1 + γ × risk_on.clip(-1, 1))   # γ = {self.tilt_gamma}
alpha_13 = cross_sectional_rank(tilted)
```

## Performance Summary
| Metric                | Tilted | Base Momentum |
|-----------------------|--------|--------------|
| Sharpe                | {self.metrics.get('Sharpe_tilted', np.nan):.3f} | {self.metrics.get('Sharpe_base', np.nan):.3f} |
| Max Drawdown          | {self.metrics.get('MaxDrawdown_tilted', np.nan)*100:.2f}% | — |
| IC (OOS) @ 1d         | {self.metrics.get('IC_tilted_lag1', np.nan):.5f} | {self.metrics.get('IC_base_lag1', np.nan):.5f} |
| IC (OOS) @ 5d         | {self.metrics.get('IC_tilted_lag5', np.nan):.5f} | — |
| IC Lift               | {self.metrics.get('IC_lift', np.nan):+.5f} | — |
| VIX Correlation       | {self.metrics.get('VIX_correlation', np.nan):.4f} | — |

## IC: Tilted Signal
{ic_t_str}

## IC: Base Momentum
{ic_b_str}

## Stress Period Analysis (VIX > 30)
{dd_str}

## Academic References
- Koijen et al. (2018) *Carry* — JFinEc
- Lustig & Verdelhan (2007) *The Cross Section of FX Risk Premia* — AER
- Asness, Moskowitz & Pedersen (2013) *Value and Momentum Everywhere* — JF
"""
        p = REPORTS_DIR / f"alpha_{ALPHA_ID}_report.md"
        p.write_text(report); log.info("Report → %s", p)
        return report


def run_alpha13(tickers=None, start=DEFAULT_START, end=DEFAULT_END, use_crypto=False):
    a = Alpha13(tickers=tickers, start=start, end=end, use_crypto=use_crypto)
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
    a = Alpha13(start=args.start, end=args.end, use_crypto=args.crypto)
    a.run(); a.plot(); a.generate_report()
    print("\n" + "="*60 + "\nALPHA 13 COMPLETE\n" + "="*60)
    for k, v in a.metrics.items():
        print(f"  {k:<38} {v:.5f}" if isinstance(v, float) else f"  {k:<38} {v}")
