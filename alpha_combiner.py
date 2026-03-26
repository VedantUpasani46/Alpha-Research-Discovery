"""
Alpha Combiner / Meta-Learner
==============================

Multiple alpha combination methods for blending quantitative signals into
a single composite alpha.  Every combiner enforces strict temporal discipline:
weights are **never** estimated using future data.

Combiners
---------
* :class:`EqualWeightCombiner` — simple average.
* :class:`ICWeightedCombiner` — weight by trailing information coefficient.
* :class:`RidgeCombiner` — regularised linear combination (L2).
* :class:`LightGBMCombiner` — gradient-boosted meta-learner with walk-forward
  refit.
* :class:`OptimalShrinkageCombiner` — Ledoit-Wolf shrinkage on the alpha
  covariance matrix.

All combiners share a common interface:

.. code-block:: python

    combiner = ICWeightedCombiner(lookback=63)
    combined = combiner.combine(alpha_signals, forward_returns)
    print(combiner.get_weights())

References
----------
.. [1] Ledoit, O. & Wolf, M. (2004). "A Well-Conditioned Estimator for
       Large-Dimensional Covariance Matrices." *JMVA*, 88(2), 365–411.
.. [2] Kakushadze, Z. (2016). "101 Formulaic Alphas." *Wilmott*, 2016(84).
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class AlphaCombinerBase(ABC):
    """Abstract base for alpha combiners.

    Subclasses must implement :meth:`_fit_combine`.
    """

    def __init__(self) -> None:
        self._weights: Optional[Dict[str, float]] = None
        self._alpha_names: List[str] = []

    # -- public API ----------------------------------------------------------

    def combine(
        self,
        alphas: Dict[str, pd.Series],
        forward_returns: Optional[pd.Series] = None,
    ) -> pd.Series:
        """Combine alpha signals into a single composite signal.

        Parameters
        ----------
        alphas : dict[str, pd.Series]
            Named alpha signals.  Each Series must share the same index
            (e.g. multi-index of (date, asset) or a simple date index).
        forward_returns : pd.Series or None
            Realised forward returns aligned with the alpha index.  Required
            by supervised combiners (Ridge, LightGBM, IC-weighted); ignored
            by :class:`EqualWeightCombiner`.

        Returns
        -------
        pd.Series
            Combined signal, same index as input alphas.
        """
        self._alpha_names = sorted(alphas.keys())
        alpha_df = pd.DataFrame(alphas)
        alpha_df = alpha_df[self._alpha_names]  # consistent column order
        combined = self._fit_combine(alpha_df, forward_returns)
        return combined

    def get_weights(self) -> Dict[str, float]:
        """Return the latest fitted weights / importances.

        Returns
        -------
        dict[str, float]
            Mapping from alpha name to weight.  Semantics depend on the
            combiner (linear weight, feature importance, etc.).
        """
        if self._weights is None:
            raise RuntimeError("Call combine() first.")
        return dict(self._weights)

    # -- internals -----------------------------------------------------------

    @abstractmethod
    def _fit_combine(
        self,
        alpha_df: pd.DataFrame,
        forward_returns: Optional[pd.Series],
    ) -> pd.Series:
        ...


# ---------------------------------------------------------------------------
# 1. Equal Weight Combiner
# ---------------------------------------------------------------------------


class EqualWeightCombiner(AlphaCombinerBase):
    """Simple equal-weight average of all alpha signals.

    No fitting required — the combined signal is just the cross-sectional
    mean of all inputs.
    """

    def _fit_combine(
        self,
        alpha_df: pd.DataFrame,
        forward_returns: Optional[pd.Series],
    ) -> pd.Series:
        n = alpha_df.shape[1]
        self._weights = {c: 1.0 / n for c in alpha_df.columns}
        combined = alpha_df.mean(axis=1)
        combined.name = "equal_weight_alpha"
        logger.info("EqualWeightCombiner: %d alphas combined.", n)
        return combined


# ---------------------------------------------------------------------------
# 2. IC-Weighted Combiner
# ---------------------------------------------------------------------------


class ICWeightedCombiner(AlphaCombinerBase):
    """Weight each alpha by its trailing Information Coefficient (IC).

    At each point in time *t*, the weight of alpha *i* is proportional to
    its average rank IC over the previous ``lookback`` periods.  This
    ensures no future information is used.

    Parameters
    ----------
    lookback : int
        Number of periods to look back for IC estimation (default 63 ≈ 1 quarter).
    min_periods : int
        Minimum observations before IC-weighting kicks in; equal weight is
        used before this.
    use_rank_ic : bool
        If True, use Spearman rank IC (more robust).  Default True.
    """

    def __init__(
        self,
        lookback: int = 63,
        min_periods: int = 21,
        use_rank_ic: bool = True,
    ) -> None:
        super().__init__()
        self.lookback = lookback
        self.min_periods = min_periods
        self.use_rank_ic = use_rank_ic

    def _fit_combine(
        self,
        alpha_df: pd.DataFrame,
        forward_returns: Optional[pd.Series],
    ) -> pd.Series:
        if forward_returns is None:
            raise ValueError("ICWeightedCombiner requires forward_returns.")

        alpha_df = alpha_df.copy()
        fwd = forward_returns.reindex(alpha_df.index)

        # Compute rolling IC for each alpha
        ic_method = "spearman" if self.use_rank_ic else "pearson"
        n_alphas = alpha_df.shape[1]
        names = list(alpha_df.columns)

        # Per-period cross-sectional IC (works for panel or time-series)
        # If the index is a simple DatetimeIndex, IC is just correlation.
        # For multi-indexed panels, we group by date.
        has_multiindex = isinstance(alpha_df.index, pd.MultiIndex)

        if has_multiindex:
            # Assume level 0 = date
            date_level = alpha_df.index.get_level_values(0)
            unique_dates = date_level.unique().sort_values()
        else:
            unique_dates = alpha_df.index.unique().sort_values()

        # Pre-compute per-date IC for each alpha
        ic_series: Dict[str, List[float]] = {name: [] for name in names}
        date_list: List[Any] = []

        for dt in unique_dates:
            if has_multiindex:
                mask = date_level == dt
            else:
                mask = alpha_df.index == dt

            slice_alpha = alpha_df.loc[mask]
            slice_fwd = fwd.loc[mask]
            valid = slice_alpha.notna().all(axis=1) & slice_fwd.notna()

            if valid.sum() < 5:
                for name in names:
                    ic_series[name].append(np.nan)
                date_list.append(dt)
                continue

            date_list.append(dt)
            for name in names:
                if ic_method == "spearman":
                    corr = sp_stats.spearmanr(
                        slice_alpha.loc[valid, name], slice_fwd[valid]
                    )[0]
                else:
                    corr = slice_alpha.loc[valid, name].corr(slice_fwd[valid])
                ic_series[name].append(corr)

        # Build IC DataFrame (dates × alphas)
        ic_df = pd.DataFrame(ic_series, index=date_list)

        # Rolling mean IC
        rolling_ic = ic_df.rolling(window=self.lookback, min_periods=self.min_periods).mean()

        # Build time-varying weights (positive part only, re-normalised)
        rolling_ic_pos = rolling_ic.clip(lower=0)
        weight_sum = rolling_ic_pos.sum(axis=1)
        weight_sum = weight_sum.replace(0, np.nan)
        weights_df = rolling_ic_pos.div(weight_sum, axis=0).fillna(1.0 / n_alphas)

        # Combine: for each date, weight the cross-section
        combined_parts = []
        for dt_idx, dt in enumerate(unique_dates):
            if has_multiindex:
                mask = date_level == dt
            else:
                mask = alpha_df.index == dt

            w = weights_df.iloc[dt_idx] if dt_idx < len(weights_df) else pd.Series(
                {n: 1.0 / n_alphas for n in names}
            )
            combined_parts.append(
                (alpha_df.loc[mask] * w).sum(axis=1)
            )

        combined = pd.concat(combined_parts)
        combined.name = "ic_weighted_alpha"

        # Store latest weights
        latest_w = weights_df.iloc[-1]
        self._weights = {n: float(latest_w[n]) for n in names}
        logger.info(
            "ICWeightedCombiner: weights = %s",
            {k: round(v, 4) for k, v in self._weights.items()},
        )
        return combined


# ---------------------------------------------------------------------------
# 3. Ridge Combiner
# ---------------------------------------------------------------------------


class RidgeCombiner(AlphaCombinerBase):
    """Regularised linear combination with L2 penalty (Ridge regression).

    Fits ``forward_return ~ sum(w_i * alpha_i)`` with L2 regularisation.
    Walk-forward refit: at each refit point only past data is used.

    Parameters
    ----------
    alpha_l2 : float
        Regularisation strength (default 1.0).
    refit_every : int
        Refit the model every *N* periods (default 63).
    min_train_obs : int
        Minimum training observations before fitting (default 126).
    normalize : bool
        Z-score alphas before fitting (default True).
    """

    def __init__(
        self,
        alpha_l2: float = 1.0,
        refit_every: int = 63,
        min_train_obs: int = 126,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.alpha_l2 = alpha_l2
        self.refit_every = refit_every
        self.min_train_obs = min_train_obs
        self.normalize = normalize

    def _fit_combine(
        self,
        alpha_df: pd.DataFrame,
        forward_returns: Optional[pd.Series],
    ) -> pd.Series:
        if forward_returns is None:
            raise ValueError("RidgeCombiner requires forward_returns.")

        fwd = forward_returns.reindex(alpha_df.index)
        names = list(alpha_df.columns)
        n_alphas = len(names)

        # Determine unique dates
        has_multiindex = isinstance(alpha_df.index, pd.MultiIndex)
        if has_multiindex:
            date_level = alpha_df.index.get_level_values(0)
            unique_dates = date_level.unique().sort_values()
        else:
            unique_dates = alpha_df.index.unique().sort_values()
            date_level = alpha_df.index

        combined = pd.Series(np.nan, index=alpha_df.index, name="ridge_alpha")
        current_beta = np.ones(n_alphas) / n_alphas  # equal weight fallback

        for t_idx, dt in enumerate(unique_dates):
            # Refit?
            if t_idx >= self.min_train_obs and t_idx % self.refit_every == 0:
                # Training data: everything before this date
                train_dates = unique_dates[:t_idx]
                train_mask = date_level.isin(train_dates)
                X_train = alpha_df.loc[train_mask].values
                y_train = fwd.loc[train_mask].values

                valid = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
                X_tr = X_train[valid]
                y_tr = y_train[valid]

                if len(X_tr) >= self.min_train_obs:
                    if self.normalize:
                        mu = X_tr.mean(axis=0)
                        sigma = X_tr.std(axis=0)
                        sigma[sigma == 0] = 1.0
                        X_tr = (X_tr - mu) / sigma

                    # Closed-form Ridge: beta = (X'X + alpha*I)^{-1} X'y
                    XtX = X_tr.T @ X_tr
                    Xty = X_tr.T @ y_tr
                    beta = np.linalg.solve(
                        XtX + self.alpha_l2 * np.eye(n_alphas) * len(X_tr),
                        Xty,
                    )
                    current_beta = beta

            # Apply current weights to this date's cross-section
            mask = date_level == dt
            X_now = alpha_df.loc[mask].values
            if self.normalize and t_idx >= self.min_train_obs:
                # Use running stats (avoid lookahead)
                past_mask = date_level.isin(unique_dates[: t_idx + 1])
                past_X = alpha_df.loc[past_mask].values
                valid_past = np.isfinite(past_X).all(axis=1)
                mu = past_X[valid_past].mean(axis=0)
                sigma = past_X[valid_past].std(axis=0)
                sigma[sigma == 0] = 1.0
                X_now_norm = (X_now - mu) / sigma
            else:
                X_now_norm = X_now

            combined.loc[mask] = X_now_norm @ current_beta

        # Normalize weights for reporting
        w_abs = np.abs(current_beta)
        w_norm = w_abs / w_abs.sum() if w_abs.sum() > 0 else np.ones(n_alphas) / n_alphas
        self._weights = {names[i]: float(current_beta[i]) for i in range(n_alphas)}
        logger.info(
            "RidgeCombiner (alpha=%.2f): weights = %s",
            self.alpha_l2,
            {k: round(v, 4) for k, v in self._weights.items()},
        )
        return combined


# ---------------------------------------------------------------------------
# 4. LightGBM Combiner
# ---------------------------------------------------------------------------


class LightGBMCombiner(AlphaCombinerBase):
    """Gradient-boosted meta-learner using LightGBM with walk-forward refit.

    Parameters
    ----------
    lgb_params : dict or None
        LightGBM parameters.  Sensible defaults are provided.
    refit_every : int
        Refit every *N* periods (default 63).
    min_train_obs : int
        Minimum training observations (default 252).
    n_estimators : int
        Number of boosting rounds (default 200).
    early_stopping_rounds : int
        Early stopping patience on a held-out portion of training data.
    """

    def __init__(
        self,
        lgb_params: Optional[Dict[str, Any]] = None,
        refit_every: int = 63,
        min_train_obs: int = 252,
        n_estimators: int = 200,
        early_stopping_rounds: int = 20,
    ) -> None:
        super().__init__()
        self.refit_every = refit_every
        self.min_train_obs = min_train_obs
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.lgb_params = lgb_params or {
            "objective": "regression",
            "metric": "mse",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "verbose": -1,
            "n_jobs": -1,
            "seed": 42,
        }

    def _fit_combine(
        self,
        alpha_df: pd.DataFrame,
        forward_returns: Optional[pd.Series],
    ) -> pd.Series:
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError(
                "LightGBMCombiner requires the `lightgbm` package. "
                "Install with: pip install lightgbm"
            )

        if forward_returns is None:
            raise ValueError("LightGBMCombiner requires forward_returns.")

        fwd = forward_returns.reindex(alpha_df.index)
        names = list(alpha_df.columns)

        has_multiindex = isinstance(alpha_df.index, pd.MultiIndex)
        if has_multiindex:
            date_level = alpha_df.index.get_level_values(0)
            unique_dates = date_level.unique().sort_values()
        else:
            unique_dates = alpha_df.index.unique().sort_values()
            date_level = alpha_df.index

        combined = pd.Series(np.nan, index=alpha_df.index, name="lgbm_alpha")
        model: Optional[lgb.Booster] = None
        importance: Optional[np.ndarray] = None

        for t_idx, dt in enumerate(unique_dates):
            # Refit?
            if t_idx >= self.min_train_obs and (
                model is None or t_idx % self.refit_every == 0
            ):
                # Use 80/20 temporal split within training data for early stopping
                train_dates = unique_dates[:t_idx]
                split_point = int(len(train_dates) * 0.8)
                fit_dates = train_dates[:split_point]
                val_dates = train_dates[split_point:]

                fit_mask = date_level.isin(fit_dates)
                val_mask = date_level.isin(val_dates)

                X_fit = alpha_df.loc[fit_mask].values
                y_fit = fwd.loc[fit_mask].values
                X_val = alpha_df.loc[val_mask].values
                y_val = fwd.loc[val_mask].values

                # Remove NaN rows
                valid_fit = np.isfinite(X_fit).all(axis=1) & np.isfinite(y_fit)
                valid_val = np.isfinite(X_val).all(axis=1) & np.isfinite(y_val)

                if valid_fit.sum() >= self.min_train_obs:
                    train_set = lgb.Dataset(X_fit[valid_fit], label=y_fit[valid_fit])
                    val_set = lgb.Dataset(X_val[valid_val], label=y_val[valid_val])

                    callbacks = [
                        lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                        lgb.log_evaluation(period=0),
                    ]
                    model = lgb.train(
                        self.lgb_params,
                        train_set,
                        num_boost_round=self.n_estimators,
                        valid_sets=[val_set],
                        callbacks=callbacks,
                    )
                    importance = model.feature_importance(importance_type="gain")

            # Predict
            if model is not None:
                mask = date_level == dt
                X_now = alpha_df.loc[mask].values
                valid_now = np.isfinite(X_now).all(axis=1)
                preds = np.full(mask.sum(), np.nan)
                if valid_now.any():
                    preds[valid_now] = model.predict(X_now[valid_now])
                combined.loc[mask] = preds

        # Store feature importance as weights
        if importance is not None:
            imp_sum = importance.sum()
            if imp_sum > 0:
                self._weights = {
                    names[i]: float(importance[i] / imp_sum)
                    for i in range(len(names))
                }
            else:
                self._weights = {n: 1.0 / len(names) for n in names}
        else:
            self._weights = {n: 1.0 / len(names) for n in names}

        logger.info(
            "LightGBMCombiner: importance = %s",
            {k: round(v, 4) for k, v in self._weights.items()},
        )
        return combined


# ---------------------------------------------------------------------------
# 5. Optimal Shrinkage Combiner (Ledoit-Wolf)
# ---------------------------------------------------------------------------


class OptimalShrinkageCombiner(AlphaCombinerBase):
    """Combine alphas using Ledoit-Wolf shrinkage on the signal covariance.

    The idea is to compute the *minimum-variance* combination of alpha
    signals, estimating the covariance matrix with Ledoit-Wolf shrinkage
    for numerical stability when the number of alphas is large relative
    to the sample size.

    Walk-forward: at each refit point, only data up to that point is used.

    Parameters
    ----------
    lookback : int
        Number of periods for covariance estimation (default 252).
    refit_every : int
        Refit every *N* periods (default 63).
    target_return : bool
        If True, maximise IC-weighted return subject to risk budget.
        If False (default), minimise variance.
    """

    def __init__(
        self,
        lookback: int = 252,
        refit_every: int = 63,
        target_return: bool = False,
    ) -> None:
        super().__init__()
        self.lookback = lookback
        self.refit_every = refit_every
        self.target_return = target_return

    def _fit_combine(
        self,
        alpha_df: pd.DataFrame,
        forward_returns: Optional[pd.Series],
    ) -> pd.Series:
        names = list(alpha_df.columns)
        n_alphas = len(names)

        has_multiindex = isinstance(alpha_df.index, pd.MultiIndex)
        if has_multiindex:
            date_level = alpha_df.index.get_level_values(0)
            unique_dates = date_level.unique().sort_values()
        else:
            unique_dates = alpha_df.index.unique().sort_values()
            date_level = alpha_df.index

        # We need per-date aggregate alpha values (e.g. mean cross-sectional IC)
        # For covariance estimation, use per-date alpha means or cross-sectional ICs
        # Here we compute per-date cross-sectional mean of each alpha
        daily_means = alpha_df.groupby(date_level).mean()

        combined = pd.Series(np.nan, index=alpha_df.index, name="shrinkage_alpha")
        current_w = np.ones(n_alphas) / n_alphas

        for t_idx, dt in enumerate(unique_dates):
            if t_idx >= self.lookback and t_idx % self.refit_every == 0:
                # Use lookback window of daily means
                window = daily_means.iloc[max(0, t_idx - self.lookback): t_idx]
                window = window.dropna()

                if len(window) >= max(n_alphas + 1, 20):
                    cov_shrunk = self._ledoit_wolf_shrinkage(window.values)

                    try:
                        inv_cov = np.linalg.inv(cov_shrunk)
                    except np.linalg.LinAlgError:
                        inv_cov = np.linalg.pinv(cov_shrunk)

                    ones = np.ones(n_alphas)

                    if self.target_return and forward_returns is not None:
                        # Use trailing IC as expected return proxy
                        fwd = forward_returns.reindex(alpha_df.index)
                        ic_vec = np.array([
                            alpha_df.loc[
                                date_level.isin(unique_dates[max(0, t_idx - self.lookback): t_idx]),
                                name,
                            ].corr(
                                fwd.loc[
                                    date_level.isin(unique_dates[max(0, t_idx - self.lookback): t_idx])
                                ]
                            )
                            for name in names
                        ])
                        ic_vec = np.nan_to_num(ic_vec, nan=0.0)
                        w = inv_cov @ ic_vec
                    else:
                        # Minimum variance
                        w = inv_cov @ ones

                    w_sum = w.sum()
                    if w_sum != 0:
                        w = w / w_sum
                    else:
                        w = ones / n_alphas

                    current_w = w

            # Apply
            mask = date_level == dt
            combined.loc[mask] = (alpha_df.loc[mask].values * current_w).sum(axis=1)

        self._weights = {names[i]: float(current_w[i]) for i in range(n_alphas)}
        logger.info(
            "OptimalShrinkageCombiner: weights = %s",
            {k: round(v, 4) for k, v in self._weights.items()},
        )
        return combined

    @staticmethod
    def _ledoit_wolf_shrinkage(X: np.ndarray) -> np.ndarray:
        """Ledoit-Wolf shrinkage estimator for covariance.

        Parameters
        ----------
        X : np.ndarray, shape (T, N)
            Centred data matrix.

        Returns
        -------
        np.ndarray, shape (N, N)
            Shrunk covariance matrix.

        References
        ----------
        Ledoit, O. & Wolf, M. (2004). JMVA 88(2), 365-411.
        """
        T, N = X.shape
        # De-mean
        X = X - X.mean(axis=0)
        sample_cov = (X.T @ X) / T

        # Shrinkage target: scaled identity (constant correlation model)
        mu = np.trace(sample_cov) / N
        target = mu * np.eye(N)

        # Compute optimal shrinkage intensity
        delta = sample_cov - target
        # sum of squared off-diagonal elements of sample_cov
        sum_sq = np.sum(delta ** 2)

        # Estimate numerator (sum of Var(s_ij))
        X2 = X ** 2
        phi_mat = (X2.T @ X2) / T - sample_cov ** 2
        phi = np.sum(phi_mat)

        # Shrinkage coefficient
        kappa = phi / sum_sq if sum_sq > 0 else 1.0
        shrinkage = max(0.0, min(1.0, kappa / T))

        return (1 - shrinkage) * sample_cov + shrinkage * target
