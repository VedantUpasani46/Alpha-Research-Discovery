"""
alpha_10_kalman_dynamic_beta.py
─────────────────────────────────
ALPHA 10 — Kalman Filter Dynamic Beta Deviation
================================================

HYPOTHESIS
----------
The Kalman filter provides the optimal estimate of the "true" time-varying
market beta for each asset.  When an asset's realized beta over a short window
EXCEEDS its Kalman-estimated beta, the asset has OVER-REACTED to recent market
moves.  This excess sensitivity tends to mean-revert back toward the Kalman
estimate.

Conversely, when the realized beta is BELOW the Kalman estimate, the asset has
under-responded and tends to subsequently move to catch up.

Signal:
    α₁₀ = -rank(β_realized - β_Kalman)

Long assets where realized beta < Kalman beta (under-responded → likely to follow).
Short assets where realized beta > Kalman beta (over-responded → will revert).

WHY KALMAN (NOT ROLLING OLS)?
──────────────────────────────
Rolling OLS assigns equal weight to all observations in the window.
The Kalman filter is a weighted smoother: it automatically discounts older
observations while incorporating our prior belief about how quickly beta changes.
The Kalman gain determines this trade-off.  A noisy state (high Q) gives higher
weight to new data; a noisy observation (high R) relies more on the prior.

KALMAN FILTER FORMULATION
--------------------------
State:        β_t (time-varying beta)
Measurement:  r_asset,t = β_t × r_market,t + ε_t

State equation:    β_t = β_{t-1} + w_t,    w_t ~ N(0, Q)  [random walk prior]
Observation eq.:   y_t = H_t × β_t + v_t,  v_t ~ N(0, R)

where H_t = r_market,t (the regressor is the market return at time t)

Predict:
    β_{t|t-1}   = β_{t-1|t-1}
    P_{t|t-1}   = P_{t-1|t-1} + Q

Update:
    K_t = P_{t|t-1} × H_t / (H_t² × P_{t|t-1} + R)   [Kalman gain]
    β_{t|t}   = β_{t|t-1} + K_t × (y_t - H_t × β_{t|t-1})
    P_{t|t}   = (1 - K_t × H_t) × P_{t|t-1}

ASSET CLASS
-----------
S&P 500 equities (or crypto basket — crypto beta w.r.t. BTC).

REBALANCE FREQUENCY
-------------------
Daily.  Beta deviations are a high-frequency signal.

VALIDATION
----------
• IC at 1-day, 5-day, 22-day
• Kalman gain convergence speed (lower gain = smoother = less noise)
• Conditional IC: stronger during high-VIX / high market-vol periods?
• Sharpe, Max Drawdown
• Distribution of beta deviations by signal quintile

REFERENCES
----------
• Kalman (1960) — A New Approach to Linear Filtering — ASME
• Faff, Hillier & Hillier (2000) — Time-varying beta risk — JBusFinAcc
• Zhou (2010) — Kalman filters for time-varying beta estimation

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
log = logging.getLogger("Alpha10")

ALPHA_ID    = "10"
ALPHA_NAME  = "Kalman_DynamicBeta"
OUTPUT_DIR  = Path("./results")
REPORTS_DIR = OUTPUT_DIR / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_START      = "2015-01-01"
DEFAULT_END        = "2024-12-31"
REALIZED_BETA_WIN  = 5            # short window for realized beta
KALMAN_Q           = 1e-4         # state noise variance (beta random walk speed)
KALMAN_R           = 1e-2         # observation noise variance
IC_LAGS            = [1, 2, 3, 5, 10, 22]
TOP_PCT            = 0.20
TC_BPS             = 8.0
IS_FRACTION        = 0.70


# ── 1D Kalman Filter for time-varying beta ────────────────────────────────────
class KalmanBetaFilter:
    """
    Single-asset 1D Kalman filter for time-varying market beta.

    Parameters
    ----------
    Q : float — state noise variance (how quickly beta can change)
                Higher Q → beta adapts faster to new data
    R : float — observation noise variance (idiosyncratic return noise)
                Higher R → less weight to new observations, smoother beta

    The ratio Q/R is the key tuning parameter.
    """

    def __init__(self, Q: float = KALMAN_Q, R: float = KALMAN_R):
        self.Q = Q   # process noise
        self.R = R   # observation noise

    def filter(
        self,
        asset_returns:  np.ndarray,  # shape (T,)
        market_returns: np.ndarray,  # shape (T,)
        beta_init:      float = 1.0,
        P_init:         float = 1.0,
    ) -> Dict[str, np.ndarray]:
        """
        Run the Kalman filter forward pass.
        Returns dict with keys:
          beta_filtered  : (T,) optimal beta estimates
          P_filtered     : (T,) estimate variance (uncertainty)
          kalman_gains   : (T,) Kalman gain at each step
          innovations    : (T,) prediction errors (y_t - H_t β_{t|t-1})
        """
        T = len(asset_returns)
        beta_filt   = np.zeros(T)
        P_filt      = np.zeros(T)
        k_gains     = np.zeros(T)
        innovations = np.zeros(T)

        beta_pred = beta_init
        P_pred    = P_init

        for t in range(T):
            H_t = market_returns[t]   # regressor (market return at t)
            y_t = asset_returns[t]    # observation (asset return at t)

            # ── Kalman Gain ───────────────────────────────────────────────
            S_t = H_t**2 * P_pred + self.R   # innovation variance
            K_t = P_pred * H_t / (S_t + 1e-12)

            # ── Innovation ────────────────────────────────────────────────
            innov_t = y_t - H_t * beta_pred

            # ── Update ────────────────────────────────────────────────────
            beta_upd = beta_pred + K_t * innov_t
            P_upd    = (1 - K_t * H_t) * P_pred

            # Store
            beta_filt[t]   = beta_upd
            P_filt[t]      = P_upd
            k_gains[t]     = K_t
            innovations[t] = innov_t

            # ── Predict next ──────────────────────────────────────────────
            beta_pred = beta_upd      # random walk: E[β_{t+1}] = β_t
            P_pred    = P_upd + self.Q

        return {
            "beta_filtered": beta_filt,
            "P_filtered":    P_filt,
            "kalman_gains":  k_gains,
            "innovations":   innovations,
        }

    def tune_hyperparams(
        self,
        asset_returns:  np.ndarray,
        market_returns: np.ndarray,
        Q_grid:         List[float] = None,
        R_grid:         List[float] = None,
    ) -> Dict[str, float]:
        """
        Grid search for Q, R that minimise cumulative squared innovation
        on a validation subsample (last 20% of data).
        """
        Q_grid = Q_grid or [1e-5, 1e-4, 1e-3, 1e-2]
        R_grid = R_grid or [1e-3, 1e-2, 1e-1, 0.5]

        T      = len(asset_returns)
        val_cut = int(T * 0.80)

        best_mse, best_Q, best_R = np.inf, self.Q, self.R
        for Q in Q_grid:
            for R in R_grid:
                kf  = KalmanBetaFilter(Q=Q, R=R)
                res = kf.filter(asset_returns, market_returns)
                # MSE on validation innovations
                val_innov = res["innovations"][val_cut:]
                mse = (val_innov**2).mean()
                if mse < best_mse:
                    best_mse, best_Q, best_R = mse, Q, R

        return {"Q": best_Q, "R": best_R, "validation_MSE": best_mse}


# ══════════════════════════════════════════════════════════════════════════════
class Alpha10:
    """
    Kalman Dynamic Beta Deviation Alpha.

    For each asset, at each time t:
      1. Estimate Kalman-filtered beta (β_Kalman)
      2. Estimate short-window rolling beta (β_realized, 5-day)
      3. Signal = -rank(β_realized - β_Kalman)
         → short over-responsive, long under-responsive assets
    """

    def __init__(
        self,
        tickers:          List[str] = None,
        start:            str       = DEFAULT_START,
        end:              str       = DEFAULT_END,
        kalman_Q:         float     = KALMAN_Q,
        kalman_R:         float     = KALMAN_R,
        realized_beta_win:int       = REALIZED_BETA_WIN,
        ic_lags:          List[int] = IC_LAGS,
        top_pct:          float     = TOP_PCT,
        tc_bps:           float     = TC_BPS,
        use_crypto:       bool      = False,
        tune_kalman:      bool      = False,
    ):
        self.tickers           = tickers or (CRYPTO_UNIVERSE[:20] if use_crypto else SP500_TICKERS[:50])
        self.start             = start
        self.end               = end
        self.kalman_Q          = kalman_Q
        self.kalman_R          = kalman_R
        self.realized_beta_win = realized_beta_win
        self.ic_lags           = ic_lags
        self.top_pct           = top_pct
        self.tc_bps            = tc_bps
        self.use_crypto        = use_crypto
        self.tune_kalman       = tune_kalman

        self._fetcher = DataFetcher()

        self.close:          Optional[pd.DataFrame] = None
        self.returns:        Optional[pd.DataFrame] = None
        self.market_ret:     Optional[pd.Series]    = None
        self.beta_kalman:    Optional[pd.DataFrame] = None
        self.beta_realized:  Optional[pd.DataFrame] = None
        self.beta_deviation: Optional[pd.DataFrame] = None
        self.kalman_gains:   Optional[pd.DataFrame] = None
        self.signals:        Optional[pd.DataFrame] = None
        self.pnl:            Optional[pd.Series]    = None
        self.ic_table:       Optional[pd.DataFrame] = None
        self.ic_is:          Optional[pd.DataFrame] = None
        self.ic_oos:         Optional[pd.DataFrame] = None
        self.vix:            Optional[pd.Series]    = None
        self.cond_ic:        Optional[pd.DataFrame] = None
        self.fm_result:      Dict                   = {}
        self.tuned_params:   Dict                   = {}
        self.metrics:        Dict                   = {}

        log.info("Alpha10 | %d tickers | %s→%s | Q=%.1e R=%.1e",
                 len(self.tickers), start, end, kalman_Q, kalman_R)

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

        # Equal-weight market return
        self.market_ret = self.returns.mean(axis=1)

        if not self.use_crypto:
            try:
                self.vix = self._fetcher.get_vix(self.start, self.end)
            except Exception:
                self.vix = None

        log.info("Loaded | %d assets | %d dates", self.close.shape[1], self.close.shape[0])

    def _compute_kalman_betas(self) -> None:
        """
        For each asset, run the Kalman filter over the full time series.
        Stores beta_kalman, kalman_gains, and optionally tunes Q/R per asset.
        """
        log.info("Computing Kalman betas for %d assets …", self.close.shape[1])
        mkt = self.market_ret.values
        beta_k_frames = {}
        gains_frames  = {}

        for ticker in self.close.columns:
            asset_ret = self.returns[ticker].values
            # align on common valid mask
            valid = np.isfinite(asset_ret) & np.isfinite(mkt)
            if valid.sum() < 30:
                continue

            Q, R = self.kalman_Q, self.kalman_R

            if self.tune_kalman and ticker in list(self.close.columns)[:3]:
                # tune only for a few assets (expensive)
                kf      = KalmanBetaFilter(Q=Q, R=R)
                params  = kf.tune_hyperparams(
                    asset_ret[valid], mkt[valid],
                    Q_grid=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
                    R_grid=[5e-3, 1e-2, 5e-2, 1e-1],
                )
                Q, R = params["Q"], params["R"]
                self.tuned_params[ticker] = params
                log.debug("Tuned %s | Q=%.1e R=%.1e MSE=%.6f", ticker, Q, R, params["validation_MSE"])

            kf  = KalmanBetaFilter(Q=Q, R=R)
            res = kf.filter(asset_ret, mkt)

            beta_k_series = pd.Series(res["beta_filtered"],
                                       index=self.returns.index,
                                       name=ticker)
            gains_series  = pd.Series(res["kalman_gains"],
                                       index=self.returns.index,
                                       name=ticker)
            beta_k_frames[ticker] = beta_k_series
            gains_frames[ticker]  = gains_series

        self.beta_kalman  = pd.DataFrame(beta_k_frames).sort_index()
        self.kalman_gains = pd.DataFrame(gains_frames).sort_index()

        log.info("Kalman betas computed | NaN fraction=%.2f%%",
                 self.beta_kalman.isna().mean().mean() * 100)
        log.info("Mean Kalman gain (convergence): %.6f",
                 self.kalman_gains.mean().mean())

    def _compute_realized_betas(self) -> None:
        """
        Short-window rolling OLS beta.
        β_realized_{t} = Cov(r_asset, r_market) / Var(r_market)
                         over last `realized_beta_win` days
        """
        log.info("Computing %d-day realized betas …", self.realized_beta_win)
        beta_r_frames = {}
        mkt = self.market_ret

        for ticker in self.beta_kalman.columns:
            asset_ret = self.returns[ticker]
            beta_series = []
            for t in range(len(self.returns)):
                if t < self.realized_beta_win:
                    beta_series.append(np.nan)
                    continue
                window_mkt   = mkt.iloc[t - self.realized_beta_win:t].values
                window_asset = asset_ret.iloc[t - self.realized_beta_win:t].values
                valid = np.isfinite(window_mkt) & np.isfinite(window_asset)
                if valid.sum() < 3:
                    beta_series.append(np.nan)
                    continue
                var_mkt = np.var(window_mkt[valid]) + 1e-12
                cov_am  = np.cov(window_asset[valid], window_mkt[valid])[0, 1]
                beta_series.append(cov_am / var_mkt)

            beta_r_frames[ticker] = pd.Series(beta_series, index=self.returns.index, name=ticker)

        self.beta_realized = pd.DataFrame(beta_r_frames).sort_index()
        log.info("Realized betas computed")

    def _compute_signal(self) -> None:
        """
        β_deviation = β_realized - β_Kalman
        α₁₀ = -rank(β_deviation)   → short over-reactive, long under-reactive
        """
        log.info("Computing signal …")
        self.beta_deviation = self.beta_realized - self.beta_kalman.reindex(
            self.beta_realized.index)
        # Winsorise at 2/98 percentile
        self.beta_deviation = self.beta_deviation.clip(
            self.beta_deviation.quantile(0.02, axis=None),
            self.beta_deviation.quantile(0.98, axis=None),
        )
        self.signals = cross_sectional_rank(-self.beta_deviation)

    def _compute_conditional_ic(self) -> None:
        """IC conditional on market vol regime (high VIX vs low VIX)."""
        if self.vix is None:
            return

        log.info("Computing regime-conditional IC …")
        fwd_5d   = self.returns.shift(-5)
        high_vix = self.vix[self.vix > 20].index
        low_vix  = self.vix[self.vix <= 20].index

        rows = []
        for name, dates in [("High VIX (>20)", high_vix), ("Low VIX (≤20)", low_vix)]:
            sigs = self.signals.loc[self.signals.index.intersection(dates)].dropna(how="all")
            fwds = fwd_5d.loc[fwd_5d.index.intersection(dates)]
            ic   = information_coefficient_matrix(sigs, fwds, [5])
            rows.append({
                "Regime": name,
                "IC_5d":  ic.loc[5, "mean_IC"] if 5 in ic.index else np.nan,
                "ICIR":   ic.loc[5, "ICIR"]    if 5 in ic.index else np.nan,
                "n":      len(dates),
            })

        self.cond_ic = pd.DataFrame(rows).set_index("Regime")
        log.info("Conditional IC:\n%s", self.cond_ic.to_string())

    def run(self) -> "Alpha10":
        self._load_data()
        self._compute_kalman_betas()
        self._compute_realized_betas()
        self._compute_signal()
        self._compute_conditional_ic()

        is_idx, oos_idx = walk_forward_split(self.close.index, IS_FRACTION)

        self.ic_table = information_coefficient_matrix(
            self.signals.dropna(how="all"), self.returns, self.ic_lags)
        self.ic_is = information_coefficient_matrix(
            self.signals.loc[self.signals.index.intersection(is_idx)].dropna(how="all"),
            self.returns.loc[self.returns.index.intersection(is_idx)], self.ic_lags)
        self.ic_oos = information_coefficient_matrix(
            self.signals.loc[self.signals.index.intersection(oos_idx)].dropna(how="all"),
            self.returns.loc[self.returns.index.intersection(oos_idx)], self.ic_lags)

        self.fm_result = fama_macbeth_regression(self.signals, self.returns, lag=1)
        log.info("FM: γ=%.5f t=%.2f", self.fm_result["gamma"], self.fm_result["t_stat"])

        self.pnl = long_short_portfolio_returns(
            self.signals.dropna(how="all"), self.returns, self.top_pct, self.tc_bps)

        self._compute_metrics()
        return self

    def _compute_metrics(self) -> None:
        pnl = self.pnl.dropna() if self.pnl is not None else pd.Series()
        ic1_is  = self.ic_is.loc[1,  "mean_IC"] if self.ic_is  is not None and 1 in self.ic_is.index  else np.nan
        ic1_oos = self.ic_oos.loc[1, "mean_IC"] if self.ic_oos is not None and 1 in self.ic_oos.index else np.nan
        ic5_oos = self.ic_oos.loc[5, "mean_IC"] if self.ic_oos is not None and 5 in self.ic_oos.index else np.nan

        mean_gain = self.kalman_gains.mean().mean() if self.kalman_gains is not None else np.nan

        self.metrics = {
            "alpha_id":            ALPHA_ID,
            "alpha_name":          ALPHA_NAME,
            "universe":            "Crypto" if self.use_crypto else "Equity",
            "n_assets":            self.close.shape[1],
            "n_dates":             self.close.shape[0],
            "kalman_Q":            self.kalman_Q,
            "kalman_R":            self.kalman_R,
            "mean_kalman_gain":    float(mean_gain),
            "IC_mean_IS_lag1":     float(ic1_is),
            "IC_mean_OOS_lag1":    float(ic1_oos),
            "IC_mean_OOS_lag5":    float(ic5_oos),
            "ICIR_IS_1d":          float(self.ic_is.loc[1,  "ICIR"]) if self.ic_is  is not None and 1 in self.ic_is.index  else np.nan,
            "ICIR_OOS_1d":         float(self.ic_oos.loc[1, "ICIR"]) if self.ic_oos is not None and 1 in self.ic_oos.index else np.nan,
            "FM_gamma":            float(self.fm_result["gamma"]),
            "FM_t_stat":           float(self.fm_result["t_stat"]),
            "Sharpe":              compute_sharpe(pnl) if len(pnl) > 0 else np.nan,
            "MaxDrawdown":         compute_max_drawdown(pnl) if len(pnl) > 0 else np.nan,
            "Annualised_Return":   float(pnl.mean() * 252) if len(pnl) > 0 else np.nan,
            "Turnover":            compute_turnover(self.signals) if self.signals is not None else np.nan,
        }
        log.info("─── Alpha 10 Metrics ────────────────────────────────")
        for k, v in self.metrics.items():
            log.info("  %-34s = %s", k, f"{v:.5f}" if isinstance(v, float) else v)

    def plot(self, save: bool = True) -> None:
        fig = plt.figure(figsize=(20, 16))
        gs  = gridspec.GridSpec(3, 2, hspace=0.42, wspace=0.30)

        # Panel 1: IC decay
        ax1 = fig.add_subplot(gs[0, 0])
        lags = [l for l in self.ic_lags if l in self.ic_table.index]
        ic_is  = [self.ic_is.loc[l,  "mean_IC"] if l in self.ic_is.index  else np.nan for l in lags]
        ic_oos = [self.ic_oos.loc[l, "mean_IC"] if l in self.ic_oos.index else np.nan for l in lags]
        ax1.plot(lags, ic_is,  "o-",  label="IS",  color="#2ca02c", lw=2)
        ax1.plot(lags, ic_oos, "s--", label="OOS", color="#d62728", lw=2)
        ax1.axhline(0, color="k", lw=0.7)
        ax1.set(xlabel="Lag (days)", ylabel="Mean IC", title="Alpha 10 — IC Decay (Beta Deviation)")
        ax1.legend(); ax1.grid(True, alpha=0.3)

        # Panel 2: Kalman vs Realized beta for one asset
        ax2 = fig.add_subplot(gs[0, 1])
        sample_tick = self.close.columns[0] if len(self.close.columns) > 0 else None
        if sample_tick and sample_tick in self.beta_kalman.columns:
            bk = self.beta_kalman[sample_tick].dropna().tail(500)
            br = self.beta_realized[sample_tick].dropna().tail(500)
            common_idx = bk.index.intersection(br.index)
            ax2.plot(common_idx, bk.loc[common_idx].values, lw=1.8, color="#1f77b4", label="Kalman β")
            ax2.plot(common_idx, br.loc[common_idx].values, lw=1.5, linestyle="--",
                     color="#ff7f0e", alpha=0.8, label=f"Realized β ({self.realized_beta_win}d)")
            ax2.fill_between(common_idx,
                             bk.loc[common_idx].values, br.loc[common_idx].values,
                             alpha=0.2, color="purple", label="Deviation (signal)")
            ax2.axhline(1.0, color="k", lw=0.6, linestyle=":", alpha=0.5)
            ax2.set(xlabel="Date", ylabel="Beta",
                    title=f"Alpha 10 — Kalman vs Realized Beta\n({sample_tick}, last 500 days)")
            ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

        # Panel 3: Kalman gain time series
        ax3 = fig.add_subplot(gs[1, 0])
        if self.kalman_gains is not None:
            mean_gains = self.kalman_gains.mean(axis=1).dropna()
            ax3.plot(mean_gains.index, mean_gains.values, lw=1.2, color="#9467bd", alpha=0.8)
            ax3.axhline(mean_gains.mean(), color="r", lw=1.0, linestyle="--",
                        label=f"Mean gain = {mean_gains.mean():.5f}")
            ax3.set(xlabel="Date", ylabel="Mean Kalman Gain",
                    title="Alpha 10 — Kalman Gain (Convergence Indicator)\n(Stable = filter has converged)")
            ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3)

        # Panel 4: Beta deviation distribution
        ax4 = fig.add_subplot(gs[1, 1])
        if self.beta_deviation is not None:
            dev_flat = self.beta_deviation.values.flatten()
            dev_flat = dev_flat[np.isfinite(dev_flat)]
            ax4.hist(dev_flat, bins=60, color="#8c564b", alpha=0.7, edgecolor="k", lw=0.4, density=True)
            ax4.axvline(0, color="r", lw=1.5, linestyle="--", label="Zero deviation")
            ax4.axvline(np.percentile(dev_flat, 80), color="green", lw=1.0, linestyle=":",
                        label="80th percentile (signal threshold)")
            x_range = np.linspace(dev_flat.min(), dev_flat.max(), 200)
            ax4.plot(x_range, sp_stats.norm.pdf(x_range, dev_flat.mean(), dev_flat.std()),
                     "k--", lw=1.2, label="Normal fit")
            ax4.set(xlabel="β_realized − β_Kalman", ylabel="Density",
                    title="Alpha 10 — Beta Deviation Distribution")
            ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3)

        # Panel 5: Cumulative PnL
        ax5 = fig.add_subplot(gs[2, :])
        if self.pnl is not None:
            cum = self.pnl.dropna().cumsum()
            roll_max = cum.cummax()
            dd = cum - roll_max
            ax5.plot(cum.index, cum.values, lw=2.0, color="#1f77b4", label="Beta Deviation L/S")
            ax5.fill_between(dd.index, dd.values, 0, where=dd.values < 0,
                             alpha=0.25, color="red", label="Drawdown")
            ax5.axhline(0, color="k", lw=0.6)
            ax5.set(title="Alpha 10 — Cumulative PnL (Net of 8 bps TC)", ylabel="Cumulative Return")
            ax5.legend(); ax5.grid(True, alpha=0.3)

        plt.suptitle(
            f"ALPHA 10 — Kalman Dynamic Beta Deviation\n"
            f"Sharpe={self.metrics.get('Sharpe', np.nan):.2f}  "
            f"IC(OOS,1d)={self.metrics.get('IC_mean_OOS_lag1', np.nan):.4f}  "
            f"IC(OOS,5d)={self.metrics.get('IC_mean_OOS_lag5', np.nan):.4f}  "
            f"FM t={self.metrics.get('FM_t_stat', np.nan):.2f}  "
            f"Mean Gain={self.metrics.get('mean_kalman_gain', np.nan):.5f}",
            fontsize=12, fontweight="bold")

        if save:
            out = REPORTS_DIR / f"alpha_{ALPHA_ID}_chart.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            log.info("Chart → %s", out)
        plt.close(fig)

    def generate_report(self) -> str:
        ic_str     = self.ic_table.reset_index().to_markdown(index=False, floatfmt=".5f") if self.ic_table is not None else "N/A"
        ic_is_str  = self.ic_is.reset_index().to_markdown(index=False, floatfmt=".5f") if self.ic_is is not None else "N/A"
        ic_oos_str = self.ic_oos.reset_index().to_markdown(index=False, floatfmt=".5f") if self.ic_oos is not None else "N/A"
        cond_str   = self.cond_ic.reset_index().to_markdown(index=False, floatfmt=".5f") if self.cond_ic is not None else "N/A"

        beta_summary = "N/A"
        if self.beta_kalman is not None:
            bs = pd.DataFrame({
                "Mean_Kalman_Beta":   self.beta_kalman.mean(),
                "Std_Kalman_Beta":    self.beta_kalman.std(),
                "Mean_Realized_Beta": self.beta_realized.mean() if self.beta_realized is not None else np.nan,
                "Mean_Deviation":     self.beta_deviation.mean() if self.beta_deviation is not None else np.nan,
            }).head(10)
            beta_summary = bs.to_markdown(floatfmt=".4f")

        report = f"""# Alpha {ALPHA_ID}: {ALPHA_NAME.replace('_', ' ')}

## Hypothesis
The Kalman filter provides the optimal estimate of an asset's time-varying
market beta.  When the short-window realized beta deviates above the Kalman
estimate, the asset has over-reacted to market moves and will revert.  Shorting
over-reactive and going long under-reactive assets generates alpha.

## Kalman Filter Specification
| Parameter   | Value       | Interpretation                         |
|-------------|-------------|----------------------------------------|
| Q (process) | {self.kalman_Q:.1e} | Beta random-walk speed (higher = faster adaptation) |
| R (obs.)    | {self.kalman_R:.1e} | Return noise (higher = smoother Kalman beta) |
| Realised window | {self.realized_beta_win}d | Short-window OLS for comparison |
| Mean Kalman Gain | {self.metrics.get('mean_kalman_gain', np.nan):.5f} | Weight on new observation |

**Kalman Gain Interpretation:** A gain near 0 means the filter relies almost entirely
on its prior estimate (very smooth beta).  A gain near 1 means beta updates fully
each day.  The optimal gain converges automatically.

## Performance Summary
| Metric               | Value |
|----------------------|-------|
| Sharpe               | {self.metrics.get('Sharpe', np.nan):.3f} |
| Annualised Return    | {self.metrics.get('Annualised_Return', np.nan)*100:.2f}% |
| Max Drawdown         | {self.metrics.get('MaxDrawdown', np.nan)*100:.2f}% |
| IC (IS)  @ 1d        | {self.metrics.get('IC_mean_IS_lag1', np.nan):.5f} |
| IC (OOS) @ 1d        | {self.metrics.get('IC_mean_OOS_lag1', np.nan):.5f} |
| IC (OOS) @ 5d        | {self.metrics.get('IC_mean_OOS_lag5', np.nan):.5f} |
| ICIR (IS)  @ 1d      | {self.metrics.get('ICIR_IS_1d', np.nan):.3f} |
| ICIR (OOS) @ 1d      | {self.metrics.get('ICIR_OOS_1d', np.nan):.3f} |
| FM γ                 | {self.metrics.get('FM_gamma', np.nan):.6f} |
| FM t-stat            | {self.metrics.get('FM_t_stat', np.nan):.3f} |
| Daily Turnover       | {self.metrics.get('Turnover', np.nan)*100:.1f}% |

## IC Decay (Full Sample)
{ic_str}

## In-Sample IC by Lag
{ic_is_str}

## Out-of-Sample IC by Lag
{ic_oos_str}

## Regime-Conditional IC
{cond_str}

## Beta Summary (Sample Assets)
{beta_summary}

## Academic References
- Kalman (1960) *A New Approach to Linear Filtering and Prediction Problems* — ASME
- Faff, Hillier & Hillier (2000) *Time Varying Beta Risk* — JBusFinAcc
- Zhou (2010) *Kalman Filters for Time-Varying Beta Estimation* — working paper
"""
        p = REPORTS_DIR / f"alpha_{ALPHA_ID}_report.md"
        p.write_text(report)
        log.info("Report → %s", p)
        return report


def run_alpha10(tickers=None, start=DEFAULT_START, end=DEFAULT_END, use_crypto=False):
    a = Alpha10(tickers=tickers, start=start, end=end, use_crypto=use_crypto)
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
    p.add_argument("--tune",   action="store_true", help="Tune Q/R via grid search")
    args = p.parse_args()
    a = Alpha10(start=args.start, end=args.end, use_crypto=args.crypto, tune_kalman=args.tune)
    a.run(); a.plot(); a.generate_report()
    print("\n" + "="*60 + "\nALPHA 10 COMPLETE\n" + "="*60)
    for k, v in a.metrics.items():
        print(f"  {k:<38} {v:.5f}" if isinstance(v, float) else f"  {k:<38} {v}")
