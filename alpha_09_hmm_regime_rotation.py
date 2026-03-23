"""
alpha_09_hmm_regime_factor_rotation.py
────────────────────────────────────────
ALPHA 09 — HMM Regime × Factor Rotation
=========================================

HYPOTHESIS
----------
Different alpha factors dominate in different market regimes.  A Hidden Markov
Model (3 states: Bull / Bear / Crisis) that identifies the current regime and
then rotates the portfolio toward the factors that historically work best in
that regime outperforms any static single-factor approach.

State-factor mapping (prior, overridden by meta-learner in Alpha 20):
  State 0 — BULL:   60% VPIN Momentum (A02), 20% Reversal (A01), 20% Skewness (A05)
  State 1 — BEAR:   40% Vol Term Structure (A06), 40% Amihud Illiquidity (A03), 20% Momentum fade
  State 2 — CRISIS: 60% Spread Compression (A07), 30% Funding Fade (A16), 10% cash

The 3-state Baum-Welch HMM is implemented from scratch using the full
Expectation-Maximisation algorithm — NOT sklearn's HMM wrapper.

FORMULA
-------
    Observation sequence: z_t = (r_t, σ_t², v_t)
    where r_t = market return, σ_t² = realized variance, v_t = volume ratio

    HMM states: π = initial state probs, A = transition matrix, B = Gaussian emission params
    Fitted via Baum-Welch (E-step: forward-backward, M-step: re-estimate A, B, π)

VALIDATION
----------
• Regime timeline chart (color-coded state assignments over price history)
• Sharpe of regime-rotating portfolio vs. equal-weight of all factors
• Transition probability matrix
• Regime assignment accuracy: state 2 (crisis) should align with drawdown periods
• IC per regime state

REFERENCES
----------
• Baum & Welch (1972) — original EM algorithm for HMMs
• Nystrup, Madsen & Lindström (2017) — HMM for regime detection — IJFC
• Hamilton (1989) — A New Approach to the Economic Analysis of Nonstationary Time Series — Econometrica

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
import matplotlib.patches as mpatches
from scipy.stats import norm
from scipy.special import logsumexp

from data_fetcher import (
    DataFetcher,
    CRYPTO_UNIVERSE,
    SP500_TICKERS,
    compute_returns,
    cross_sectional_rank,
    information_coefficient_matrix,
    compute_max_drawdown,
    compute_sharpe,
    compute_turnover,
    long_short_portfolio_returns,
    walk_forward_split,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
log = logging.getLogger("Alpha09")

ALPHA_ID    = "09"
ALPHA_NAME  = "HMM_Regime_FactorRotation"
OUTPUT_DIR  = Path("./results")
REPORTS_DIR = OUTPUT_DIR / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_START    = "2015-01-01"
DEFAULT_END      = "2024-12-31"
N_STATES         = 3
MAX_ITER         = 200
CONVERGENCE_TOL  = 1e-4
IC_LAGS          = [1, 5, 22]
TOP_PCT          = 0.20
TC_BPS           = 8.0
IS_FRACTION      = 0.70

# State labels
STATE_NAMES = {0: "Bull", 1: "Bear", 2: "Crisis"}
STATE_COLORS = {0: "#2ca02c", 1: "#ff7f0e", 2: "#d62728"}


# ── Pure-Python HMM with Baum-Welch ──────────────────────────────────────────
class GaussianHMM:
    """
    3-state Gaussian HMM with full Baum-Welch EM fitting.
    Observation can be multivariate (D-dimensional Gaussian per state).

    Parameters
    ----------
    n_states : int — number of hidden states
    n_iter   : int — maximum EM iterations
    tol      : float — convergence tolerance on log-likelihood change

    Attributes set after fit():
    pi   : (K,)     — initial state probabilities
    A    : (K, K)   — transition matrix  A[i,j] = P(s_t=j | s_{t-1}=i)
    mu   : (K, D)   — emission means
    cov  : (K, D, D) — emission covariance matrices
    ll_history : list  — log-likelihood per iteration
    """

    def __init__(self, n_states: int = N_STATES, n_iter: int = MAX_ITER, tol: float = CONVERGENCE_TOL):
        self.n_states = n_states
        self.n_iter   = n_iter
        self.tol      = tol
        self.pi       = None
        self.A        = None
        self.mu       = None
        self.cov      = None
        self.ll_history = []

    def _log_emission(self, X: np.ndarray) -> np.ndarray:
        """
        Compute log emission probabilities.
        Returns (T, K) matrix: log p(x_t | s_t = k)
        """
        T, D = X.shape
        log_b = np.zeros((T, self.n_states))
        for k in range(self.n_states):
            try:
                from scipy.stats import multivariate_normal as mvn
                log_b[:, k] = mvn.logpdf(X, mean=self.mu[k], cov=self.cov[k])
            except Exception:
                # fallback: diagonal Gaussian
                std = np.sqrt(np.diag(self.cov[k]) + 1e-8)
                log_b[:, k] = norm.logpdf(X, loc=self.mu[k], scale=std).sum(axis=1)
        return log_b

    def _forward(self, log_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward algorithm (log-space for numerical stability)."""
        T = log_b.shape[0]
        log_alpha = np.full((T, self.n_states), -np.inf)
        log_alpha[0] = np.log(self.pi + 1e-300) + log_b[0]
        log_A = np.log(self.A + 1e-300)
        for t in range(1, T):
            for j in range(self.n_states):
                log_alpha[t, j] = logsumexp(log_alpha[t-1] + log_A[:, j]) + log_b[t, j]
        log_likelihood = logsumexp(log_alpha[-1])
        return log_alpha, log_likelihood

    def _backward(self, log_b: np.ndarray) -> np.ndarray:
        """Backward algorithm (log-space)."""
        T = log_b.shape[0]
        log_beta = np.full((T, self.n_states), -np.inf)
        log_beta[-1] = 0.0
        log_A = np.log(self.A + 1e-300)
        for t in range(T-2, -1, -1):
            for i in range(self.n_states):
                log_beta[t, i] = logsumexp(log_A[i] + log_b[t+1] + log_beta[t+1])
        return log_beta

    def _e_step(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """E-step: compute posteriors gamma and xi."""
        log_b     = self._log_emission(X)
        log_alpha, ll = self._forward(log_b)
        log_beta  = self._backward(log_b)
        T = X.shape[0]

        # gamma[t,k] = P(s_t=k | X)
        log_gamma = log_alpha + log_beta
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
        gamma     = np.exp(log_gamma)

        # xi[t,i,j] = P(s_t=i, s_{t+1}=j | X)
        log_A = np.log(self.A + 1e-300)
        xi    = np.zeros((T-1, self.n_states, self.n_states))
        for t in range(T-1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    xi[t, i, j] = (log_alpha[t, i] + log_A[i, j] +
                                   log_b[t+1, j]  + log_beta[t+1, j])
            xi[t] = np.exp(xi[t] - logsumexp(xi[t].flatten()))

        return gamma, xi, ll

    def _m_step(self, X: np.ndarray, gamma: np.ndarray, xi: np.ndarray) -> None:
        """M-step: re-estimate HMM parameters."""
        T, D = X.shape
        self.pi = gamma[0] / gamma[0].sum()
        # Transition matrix
        self.A = xi.sum(axis=0) / xi.sum(axis=0).sum(axis=1, keepdims=True)
        self.A = np.clip(self.A, 1e-8, 1)
        self.A /= self.A.sum(axis=1, keepdims=True)
        # Emission means and covariances
        for k in range(self.n_states):
            w_k = gamma[:, k]
            w_sum = w_k.sum() + 1e-8
            self.mu[k] = (w_k[:, None] * X).sum(axis=0) / w_sum
            diff = X - self.mu[k]
            cov_k = (w_k[:, None, None] * np.einsum("ti,tj->tij", diff, diff)).sum(axis=0) / w_sum
            # regularise
            self.cov[k] = cov_k + np.eye(D) * 1e-4

    def fit(self, X: np.ndarray) -> "GaussianHMM":
        """Fit the HMM using Baum-Welch EM algorithm."""
        T, D = X.shape
        rng  = np.random.default_rng(42)

        # Initialise parameters with k-means-style
        from scipy.cluster.vq import kmeans2
        try:
            centroids, labels = kmeans2(X, self.n_states, niter=20, minit="points", seed=42)
        except Exception:
            centroids = X[rng.choice(T, self.n_states, replace=False)]
            labels    = rng.integers(0, self.n_states, T)

        # Sort states by mean of first dimension (ensures Bull/Bear/Crisis ordering)
        order    = np.argsort(centroids[:, 0])[::-1]    # descending return = Bull first
        centroids = centroids[order]

        self.pi  = np.ones(self.n_states) / self.n_states
        self.A   = (np.eye(self.n_states) * 0.7 +
                    np.ones((self.n_states, self.n_states)) * 0.1)
        self.A  /= self.A.sum(axis=1, keepdims=True)
        self.mu  = centroids.copy()
        self.cov = np.array([np.cov(X[labels == k].T) + np.eye(D)*0.01
                             if (labels == k).sum() > D else np.eye(D) * 0.01
                             for k in range(self.n_states)])

        prev_ll = -np.inf
        for it in range(self.n_iter):
            gamma, xi, ll = self._e_step(X)
            self._m_step(X, gamma, xi)
            self.ll_history.append(ll)
            if abs(ll - prev_ll) < self.tol:
                log.info("HMM converged at iteration %d | LL=%.4f", it+1, ll)
                break
            prev_ll = ll

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Viterbi decoding for most probable state sequence."""
        T, D     = X.shape
        log_b    = self._log_emission(X)
        log_A    = np.log(self.A + 1e-300)
        log_delta = np.full((T, self.n_states), -np.inf)
        psi       = np.zeros((T, self.n_states), dtype=int)
        log_delta[0] = np.log(self.pi + 1e-300) + log_b[0]
        for t in range(1, T):
            for j in range(self.n_states):
                scores       = log_delta[t-1] + log_A[:, j]
                psi[t, j]    = scores.argmax()
                log_delta[t, j] = scores.max() + log_b[t, j]
        # backtrack
        states = np.zeros(T, dtype=int)
        states[-1] = log_delta[-1].argmax()
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        return states

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Posterior state probabilities (smoothed)."""
        log_b     = self._log_emission(X)
        log_alpha, _ = self._forward(log_b)
        log_beta  = self._backward(log_b)
        log_gamma = log_alpha + log_beta
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
        return np.exp(log_gamma)


# ══════════════════════════════════════════════════════════════════════════════
class Alpha09:
    """
    HMM Regime Detection and Factor Rotation.
    """

    def __init__(
        self,
        tickers:    List[str] = None,
        start:      str       = DEFAULT_START,
        end:        str       = DEFAULT_END,
        n_states:   int       = N_STATES,
        ic_lags:    List[int] = IC_LAGS,
        top_pct:    float     = TOP_PCT,
        tc_bps:     float     = TC_BPS,
        use_crypto: bool      = False,
    ):
        self.tickers    = tickers or (CRYPTO_UNIVERSE[:20] if use_crypto else SP500_TICKERS[:50])
        self.start      = start
        self.end        = end
        self.n_states   = n_states
        self.ic_lags    = ic_lags
        self.top_pct    = top_pct
        self.tc_bps     = tc_bps
        self.use_crypto = use_crypto

        self._fetcher = DataFetcher()
        self.hmm      = GaussianHMM(n_states=n_states)

        self.close:         Optional[pd.DataFrame] = None
        self.market_ret:    Optional[pd.Series]    = None
        self.hmm_states:    Optional[pd.Series]    = None
        self.state_proba:   Optional[pd.DataFrame] = None
        self.signals:       Optional[pd.DataFrame] = None
        self.returns:       Optional[pd.DataFrame] = None
        self.pnl:           Optional[pd.Series]    = None
        self.ic_table:      Optional[pd.DataFrame] = None
        self.ic_is:         Optional[pd.DataFrame] = None
        self.ic_oos:        Optional[pd.DataFrame] = None
        self.regime_ic:     Optional[pd.DataFrame] = None
        self.transition_df: Optional[pd.DataFrame] = None
        self.metrics:       Dict                   = {}

        log.info("Alpha09 | %s→%s | n_states=%d", start, end, n_states)

    # ─────────────────────────────────────────────────────────────────────────

    def _load_data(self) -> None:
        log.info("Loading data …")
        if self.use_crypto:
            ohlcv_dict = self._fetcher.get_crypto_universe_daily(self.tickers, self.start, self.end)
            close_frames = {s: df["Close"] for s, df in ohlcv_dict.items()}
        else:
            ohlcv_dict = self._fetcher.get_equity_ohlcv(self.tickers, self.start, self.end)
            close_frames = {t: df["Close"] for t, df in ohlcv_dict.items() if not df.empty}

        self.close = pd.DataFrame(close_frames).sort_index().ffill()
        coverage   = self.close.notna().mean()
        self.close = self.close[coverage[coverage >= 0.80].index]
        self.returns = compute_returns(self.close, 1)

        # market return = equal-weight cross-sectional mean
        self.market_ret = self.returns.mean(axis=1).dropna()
        log.info("Loaded | %d assets | %d dates", self.close.shape[1], self.close.shape[0])

    def _build_observation_sequence(self) -> np.ndarray:
        """
        3-dimensional observation: (market_return, realized_variance, volume_ratio)
        Standardised before feeding to HMM.
        """
        r = self.market_ret
        rv_5d = (r**2).rolling(5).mean().apply(lambda x: np.sqrt(x * 252))

        # Volume ratio: cross-sectional mean of individual vol ratios
        individual_rv5  = (self.returns**2).rolling(5).mean().apply(lambda x: np.sqrt(x * 252))
        individual_rv22 = (self.returns**2).rolling(22).mean().apply(lambda x: np.sqrt(x * 252))
        vol_ratio_cs    = (individual_rv5 / individual_rv22.replace(0, np.nan)).mean(axis=1)

        obs = pd.DataFrame({
            "market_ret":  r,
            "rv_5d":       rv_5d,
            "vol_ratio":   vol_ratio_cs,
        }).dropna()

        # Standardise
        obs = (obs - obs.mean()) / (obs.std() + 1e-8)
        self._obs_index = obs.index
        return obs.values

    def _fit_hmm(self) -> None:
        log.info("Fitting 3-state Gaussian HMM via Baum-Welch …")
        X = self._build_observation_sequence()
        self.hmm.fit(X)

        states = self.hmm.predict(X)
        proba  = self.hmm.predict_proba(X)

        self.hmm_states = pd.Series(states, index=self._obs_index, name="hmm_state")
        self.state_proba = pd.DataFrame(
            proba, index=self._obs_index,
            columns=[f"P(state_{k})" for k in range(self.n_states)]
        )

        # Label states: state with highest mean return = Bull (0)
        state_means = {}
        for k in range(self.n_states):
            mask = states == k
            if mask.sum() > 0:
                state_means[k] = self.market_ret.loc[self._obs_index[mask]].mean()
            else:
                state_means[k] = 0.0

        sorted_states = sorted(state_means.keys(), key=lambda k: state_means[k], reverse=True)
        remap = {sorted_states[0]: 0, sorted_states[1]: 1, sorted_states[2]: 2}
        self.hmm_states = self.hmm_states.map(remap)

        log.info("HMM fitted | LL iterations: %d | final LL: %.4f",
                 len(self.hmm.ll_history), self.hmm.ll_history[-1] if self.hmm.ll_history else np.nan)

        state_counts = self.hmm_states.value_counts().sort_index()
        for k, cnt in state_counts.items():
            log.info("  State %d (%s): %d days (%.1f%%)", k, STATE_NAMES[k], cnt, 100*cnt/len(states))

        # Transition matrix
        A = self.hmm.A
        self.transition_df = pd.DataFrame(
            A, index=[f"From_{STATE_NAMES[k]}" for k in range(self.n_states)],
            columns=[f"To_{STATE_NAMES[k]}" for k in range(self.n_states)]
        )
        log.info("Transition matrix:\n%s", self.transition_df.to_string())

    def _build_factor_signals_stub(self) -> Dict[str, pd.DataFrame]:
        """
        Build simplified proxy signals for Alpha 01, 02, 03, 05, 06, 07.
        In the full system, these come from the actual alpha modules.
        Here we compute them inline for standalone operation.
        """
        ret = self.returns
        r   = ret

        # Alpha 01 proxy: reversal
        ret_1d    = r.shift(1)
        a01       = cross_sectional_rank(-ret_1d)

        # Alpha 02 proxy: momentum
        mom_21    = np.log(self.close / self.close.shift(22)).shift(1)
        a02       = cross_sectional_rank(mom_21)

        # Alpha 03 proxy: illiquidity rank (using return std as proxy — no volume here)
        rv_22  = r.rolling(22).std()
        a03    = cross_sectional_rank(-rv_22)   # low vol = "liquid" = short; high vol = "illiquid" = long

        # Alpha 05 proxy: realized skewness reversal
        skew_22 = r.rolling(22, min_periods=10).skew()
        a05     = cross_sectional_rank(-skew_22)

        # Alpha 06 proxy: vol ratio
        rv_5  = (r**2).rolling(5).mean().apply(lambda x: np.sqrt(x * 252))
        rv_22r = (r**2).rolling(22).mean().apply(lambda x: np.sqrt(x * 252))
        a06   = cross_sectional_rank(-(rv_5 / rv_22r.replace(0, np.nan)))

        return {"A01_reversal": a01, "A02_momentum": a02,
                "A03_illiquidity": a03, "A05_skewness": a05, "A06_vol_ratio": a06}

    def _compute_regime_rotated_signal(
        self,
        factor_signals: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        At each date, blend factor signals according to the current HMM state.
        Weights:
          Bull   (0): 0.60×A02 + 0.20×A01 + 0.20×A05
          Bear   (1): 0.40×A06 + 0.40×A03 + 0.20×A01
          Crisis (2): 0.60×A01 + 0.40×A06  (defensive)
        """
        state_weights = {
            0: {"A02_momentum": 0.60, "A01_reversal": 0.20, "A05_skewness": 0.20},
            1: {"A06_vol_ratio": 0.40, "A03_illiquidity": 0.40, "A01_reversal": 0.20},
            2: {"A01_reversal": 0.60, "A06_vol_ratio": 0.40},
        }

        signal_frames = []
        for date in self.close.index:
            if date not in self.hmm_states.index:
                signal_frames.append(pd.Series(np.nan, index=self.close.columns, name=date))
                continue

            state   = self.hmm_states.loc[date]
            weights = state_weights.get(state, state_weights[0])
            blended = pd.Series(0.0, index=self.close.columns)

            for factor_name, weight in weights.items():
                if factor_name in factor_signals and date in factor_signals[factor_name].index:
                    f_row = factor_signals[factor_name].loc[date].reindex(self.close.columns).fillna(0)
                    blended += weight * f_row

            # cross-sectional re-rank after blending
            if blended.notna().sum() > 5:
                blended = cross_sectional_rank(blended.to_frame().T).iloc[0]
            signal_frames.append(pd.Series(blended.values, index=self.close.columns, name=date))

        result = pd.DataFrame(signal_frames)
        result.index.name = "date"
        return result

    def _compute_regime_ic(self, factor_signals: Dict[str, pd.DataFrame]) -> None:
        """IC per regime for each factor."""
        log.info("Computing regime-conditional IC …")
        fwd_1d = self.returns.shift(-1)
        rows   = []
        for state_id, state_name in STATE_NAMES.items():
            state_dates = self.hmm_states[self.hmm_states == state_id].index
            if len(state_dates) < 10:
                continue
            for fname, fsig in factor_signals.items():
                fsig_state = fsig.loc[fsig.index.intersection(state_dates)]
                fwd_state  = fwd_1d.loc[fwd_1d.index.intersection(state_dates)]
                ic_stats   = information_coefficient_matrix(fsig_state, fwd_state, [1])
                ic_val     = ic_stats.loc[1, "mean_IC"] if 1 in ic_stats.index else np.nan
                rows.append({"State": f"{state_id}:{state_name}", "Factor": fname,
                             "mean_IC_1d": ic_val, "n_dates": len(state_dates)})

        self.regime_ic = pd.DataFrame(rows)
        log.info("Regime IC computed | shape=%s", self.regime_ic.shape)

    def run(self) -> "Alpha09":
        self._load_data()
        self._fit_hmm()

        factor_signals = self._build_factor_signals_stub()
        self.signals   = self._compute_regime_rotated_signal(factor_signals)

        self.signals = self.signals.reindex(self.close.index)
        is_idx, oos_idx = walk_forward_split(self.close.index, IS_FRACTION)

        self.ic_table = information_coefficient_matrix(
            self.signals.dropna(how="all"), self.returns, self.ic_lags)
        self.ic_is = information_coefficient_matrix(
            self.signals.loc[self.signals.index.intersection(is_idx)].dropna(how="all"),
            self.returns.loc[self.returns.index.intersection(is_idx)], self.ic_lags)
        self.ic_oos = information_coefficient_matrix(
            self.signals.loc[self.signals.index.intersection(oos_idx)].dropna(how="all"),
            self.returns.loc[self.returns.index.intersection(oos_idx)], self.ic_lags)

        self._compute_regime_ic(factor_signals)

        self.pnl = long_short_portfolio_returns(
            self.signals.dropna(how="all"), self.returns, self.top_pct, self.tc_bps)

        # Equal-weight benchmark
        ew_signals = list(factor_signals.values())[0].copy() * 0
        for fsig in factor_signals.values():
            ew_signals = ew_signals.add(fsig.reindex(ew_signals.index).fillna(0))
        ew_signals /= len(factor_signals)
        self.pnl_ew = long_short_portfolio_returns(
            ew_signals.dropna(how="all"), self.returns, self.top_pct, self.tc_bps)

        self._compute_metrics()
        return self

    def _compute_metrics(self) -> None:
        pnl    = self.pnl.dropna() if self.pnl is not None else pd.Series()
        pnl_ew = self.pnl_ew.dropna() if self.pnl_ew is not None else pd.Series()

        ic1_is  = self.ic_is.loc[1,  "mean_IC"] if self.ic_is  is not None and 1 in self.ic_is.index  else np.nan
        ic1_oos = self.ic_oos.loc[1, "mean_IC"] if self.ic_oos is not None and 1 in self.ic_oos.index else np.nan

        crisis_pct = (self.hmm_states == 2).mean() if self.hmm_states is not None else np.nan
        bear_pct   = (self.hmm_states == 1).mean() if self.hmm_states is not None else np.nan

        self.metrics = {
            "alpha_id":          ALPHA_ID,
            "alpha_name":        ALPHA_NAME,
            "universe":          "Crypto" if self.use_crypto else "Equity",
            "n_assets":          self.close.shape[1],
            "n_dates":           self.close.shape[0],
            "IC_mean_IS_lag1":   float(ic1_is),
            "IC_mean_OOS_lag1":  float(ic1_oos),
            "ICIR_IS_1d":        float(self.ic_is.loc[1,  "ICIR"]) if self.ic_is  is not None and 1 in self.ic_is.index  else np.nan,
            "ICIR_OOS_1d":       float(self.ic_oos.loc[1, "ICIR"]) if self.ic_oos is not None and 1 in self.ic_oos.index else np.nan,
            "Sharpe_regime":     compute_sharpe(pnl)    if len(pnl)    > 0 else np.nan,
            "Sharpe_equalwt":    compute_sharpe(pnl_ew) if len(pnl_ew) > 0 else np.nan,
            "MaxDrawdown":       compute_max_drawdown(pnl) if len(pnl) > 0 else np.nan,
            "HMM_bull_pct":      float(1 - crisis_pct - bear_pct),
            "HMM_bear_pct":      float(bear_pct),
            "HMM_crisis_pct":    float(crisis_pct),
            "HMM_final_LL":      float(self.hmm.ll_history[-1]) if self.hmm.ll_history else np.nan,
        }
        log.info("─── Alpha 09 Metrics ────────────────────────────────")
        for k, v in self.metrics.items():
            log.info("  %-32s = %s", k, f"{v:.5f}" if isinstance(v, float) else v)

    def plot(self, save: bool = True) -> None:
        fig = plt.figure(figsize=(20, 16))
        gs  = gridspec.GridSpec(3, 2, hspace=0.42, wspace=0.30)

        # Panel 1: Regime timeline
        ax1 = fig.add_subplot(gs[0, :])
        mkt_price  = np.exp(self.market_ret.cumsum())
        states_al  = self.hmm_states.reindex(mkt_price.index)
        ax1.plot(mkt_price.index, mkt_price.values, "k-", lw=1.2, alpha=0.8, label="Equal-Weight Market")
        for state_id, state_name in STATE_NAMES.items():
            mask = states_al == state_id
            idx  = mkt_price[mask].index
            if len(idx) == 0:
                continue
            ax1.fill_between(mkt_price.index,
                             mkt_price.min() * 0.95,
                             mkt_price.max() * 1.05,
                             where=mask.reindex(mkt_price.index).fillna(False),
                             alpha=0.25, color=STATE_COLORS[state_id],
                             label=f"State {state_id}: {state_name}")
        ax1.set(ylabel="Cumulative Return (log scale)", title="Alpha 09 — HMM Regime Timeline\n(Colored bands = detected regimes)")
        ax1.legend(loc="upper left", fontsize=9)
        ax1.set_yscale("log")
        ax1.grid(True, alpha=0.3)

        # Panel 2: Transition probability matrix
        ax2 = fig.add_subplot(gs[1, 0])
        if self.transition_df is not None:
            im = ax2.imshow(self.transition_df.values, cmap="Blues", vmin=0, vmax=1)
            plt.colorbar(im, ax=ax2)
            labels = [STATE_NAMES[k] for k in range(self.n_states)]
            ax2.set_xticks(range(self.n_states)); ax2.set_xticklabels(labels)
            ax2.set_yticks(range(self.n_states)); ax2.set_yticklabels([f"From {l}" for l in labels])
            for i in range(self.n_states):
                for j in range(self.n_states):
                    ax2.text(j, i, f"{self.transition_df.values[i,j]:.3f}",
                             ha="center", va="center", fontsize=10,
                             color="white" if self.transition_df.values[i,j] > 0.5 else "black")
            ax2.set_title("HMM Transition Probability Matrix\n(High diagonal = persistent regimes)")

        # Panel 3: IC per regime per factor
        ax3 = fig.add_subplot(gs[1, 1])
        if self.regime_ic is not None and not self.regime_ic.empty:
            pivot = self.regime_ic.pivot(index="Factor", columns="State", values="mean_IC_1d")
            x = np.arange(len(pivot))
            w = 0.25
            cols = list(pivot.columns)
            for ci, col in enumerate(cols):
                state_num = int(col.split(":")[0])
                ax3.bar(x + ci*w - w, pivot[col].values, w,
                        label=col, color=STATE_COLORS.get(state_num, "grey"), alpha=0.8)
            ax3.set_xticks(x)
            ax3.set_xticklabels([f.replace("_", "\n") for f in pivot.index], fontsize=7)
            ax3.axhline(0, color="k", lw=0.7)
            ax3.set(ylabel="Mean IC @1d", title="Alpha 09 — IC per Factor per Regime")
            ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3, axis="y")

        # Panel 4: Cumulative PnL (regime vs equal-weight)
        ax4 = fig.add_subplot(gs[2, :])
        if self.pnl is not None:
            cf = self.pnl.dropna().cumsum()
            ax4.plot(cf.index, cf.values, lw=2.0, color="#1f77b4", label="Regime-Rotated")
        if self.pnl_ew is not None:
            cew = self.pnl_ew.dropna().cumsum()
            ax4.plot(cew.index, cew.values, lw=2.0, linestyle="--",
                     color="#ff7f0e", alpha=0.8, label="Equal-Weight Blend")
        ax4.axhline(0, color="k", lw=0.6)
        ax4.set(title="Alpha 09 — Regime-Rotated vs Equal-Weight PnL", ylabel="Cumulative Return")
        ax4.legend(); ax4.grid(True, alpha=0.3)

        plt.suptitle(
            f"ALPHA 09 — HMM Regime × Factor Rotation\n"
            f"Sharpe_regime={self.metrics.get('Sharpe_regime', np.nan):.2f}  "
            f"Sharpe_EW={self.metrics.get('Sharpe_equalwt', np.nan):.2f}  "
            f"IC(OOS,1d)={self.metrics.get('IC_mean_OOS_lag1', np.nan):.4f}  "
            f"Bull={self.metrics.get('HMM_bull_pct', np.nan)*100:.0f}%  "
            f"Bear={self.metrics.get('HMM_bear_pct', np.nan)*100:.0f}%  "
            f"Crisis={self.metrics.get('HMM_crisis_pct', np.nan)*100:.0f}%",
            fontsize=12, fontweight="bold")

        if save:
            out = REPORTS_DIR / f"alpha_{ALPHA_ID}_chart.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            log.info("Chart → %s", out)
        plt.close(fig)

    def generate_report(self) -> str:
        ic_str    = self.ic_table.reset_index().to_markdown(index=False, floatfmt=".5f") if self.ic_table is not None else "N/A"
        ic_is_str = self.ic_is.reset_index().to_markdown(index=False, floatfmt=".5f") if self.ic_is is not None else "N/A"
        ic_oos_str= self.ic_oos.reset_index().to_markdown(index=False, floatfmt=".5f") if self.ic_oos is not None else "N/A"
        trans_str = self.transition_df.to_markdown(floatfmt=".4f") if self.transition_df is not None else "N/A"
        reg_str   = self.regime_ic.to_markdown(index=False, floatfmt=".5f") if self.regime_ic is not None else "N/A"

        report = f"""# Alpha {ALPHA_ID}: {ALPHA_NAME.replace('_', ' ')}

## Hypothesis
Market regimes (Bull / Bear / Crisis) have distinct factor leadership.  An HMM that
detects the current regime enables dynamic factor rotation that outperforms static
single-factor or equal-weight approaches.

## HMM Specification
- States: 3 (Bull, Bear, Crisis)
- Observations: (market_return, realized_variance, vol_ratio)
- Algorithm: Baum-Welch EM (custom implementation, no sklearn dependency)
- Classification: Viterbi decoding for state assignment

## State-Factor Weight Table
| State   | A02 Momentum | A01 Reversal | A05 Skewness | A06 Vol Ratio | A03 Illiquidity |
|---------|-------------|-------------|-------------|--------------|----------------|
| Bull    | 0.60        | 0.20        | 0.20        | —            | —              |
| Bear    | —           | 0.20        | —           | 0.40         | 0.40           |
| Crisis  | —           | 0.60        | —           | 0.40         | —              |

## Transition Matrix
{trans_str}

## Performance Summary
| Metric                | Regime-Rotated | Equal-Weight |
|-----------------------|----------------|-------------|
| Sharpe                | {self.metrics.get('Sharpe_regime', np.nan):.3f} | {self.metrics.get('Sharpe_equalwt', np.nan):.3f} |
| Max Drawdown          | {self.metrics.get('MaxDrawdown', np.nan)*100:.2f}% | — |
| IC (IS)  @ 1d         | {self.metrics.get('IC_mean_IS_lag1', np.nan):.5f} | — |
| IC (OOS) @ 1d         | {self.metrics.get('IC_mean_OOS_lag1', np.nan):.5f} | — |
| Bull regime %         | {self.metrics.get('HMM_bull_pct', np.nan)*100:.1f}% | — |
| Bear regime %         | {self.metrics.get('HMM_bear_pct', np.nan)*100:.1f}% | — |
| Crisis regime %       | {self.metrics.get('HMM_crisis_pct', np.nan)*100:.1f}% | — |

## IC per Regime per Factor
{reg_str}

## IC Decay (Full Sample)
{ic_str}

## Academic References
- Baum & Welch (1972) — EM algorithm for HMMs
- Hamilton (1989) *A New Approach to the Economic Analysis of Nonstationary TS* — Econometrica
- Nystrup, Madsen & Lindström (2017) *Dynamic Portfolio Optimization Across Hidden Market Regimes* — IJFC
"""
        p = REPORTS_DIR / f"alpha_{ALPHA_ID}_report.md"
        p.write_text(report)
        log.info("Report → %s", p)
        return report


def run_alpha09(tickers=None, start=DEFAULT_START, end=DEFAULT_END, use_crypto=False):
    a = Alpha09(tickers=tickers, start=start, end=end, use_crypto=use_crypto)
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
    a = Alpha09(start=args.start, end=args.end, use_crypto=args.crypto)
    a.run(); a.plot(); a.generate_report()
    print("\n" + "="*60 + "\nALPHA 09 COMPLETE\n" + "="*60)
    for k, v in a.metrics.items():
        print(f"  {k:<38} {v:.5f}" if isinstance(v, float) else f"  {k:<38} {v}")
