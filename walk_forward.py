"""
Walk-Forward Validation Engine
==============================

Production-quality rolling walk-forward cross-validation for alpha signals.

This module implements *genuine* walk-forward (rolling / expanding window)
cross-validation with embargo and purging — NOT a naive 70/30 train/test
split.  Every fold honours temporal ordering so that no future information
can leak into the training set.

Key features
------------
* Rolling **and** expanding window modes.
* Embargo period between train and test sets to prevent information leakage
  from autocorrelated features (López de Prado 2018, Ch. 7).
* Purging of overlapping samples that straddle the train/test boundary.
* Per-fold statistics: IC, rank IC, annualised Sharpe, max drawdown, turnover.
* Aggregate statistics with standard errors and confidence intervals.
* Deflated Sharpe Ratio (Bailey & López de Prado 2014).
* Publication-quality cumulative OOS performance plot.

References
----------
.. [1] López de Prado, M. (2018). *Advances in Financial Machine Learning*.
       Wiley. Chapter 7 — Cross-Validation in Finance.
.. [2] Bailey, D. H. & López de Prado, M. (2014). "The Deflated Sharpe
       Ratio: Correcting for Selection Bias, Backtest Over-fitting, and
       Non-Normality." *Journal of Portfolio Management*, 40(5), 94–107.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper types
# ---------------------------------------------------------------------------

IndexPair = Tuple[np.ndarray, np.ndarray]  # (train_indices, test_indices)


class WindowMode(str, Enum):
    """Window mode for walk-forward validation."""

    ROLLING = "rolling"
    EXPANDING = "expanding"


# ---------------------------------------------------------------------------
# Fold-level statistics
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    """Statistics for a single walk-forward fold.

    Attributes
    ----------
    fold_id : int
        Zero-based fold index.
    train_start / train_end : pd.Timestamp
        Inclusive bounds of the training window.
    test_start / test_end : pd.Timestamp
        Inclusive bounds of the test window.
    ic : float
        Pearson Information Coefficient (signal vs forward return).
    rank_ic : float
        Spearman rank IC.
    sharpe : float
        Annualised Sharpe ratio of the signal-weighted returns in the test
        window (assuming 252 trading days / year).
    max_drawdown : float
        Maximum drawdown in the test window (as a positive fraction).
    turnover : float
        Mean single-period turnover (fraction of portfolio traded).
    n_train : int
        Number of training observations.
    n_test : int
        Number of test observations.
    oos_returns : pd.Series
        Daily out-of-sample returns for this fold.
    """

    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    ic: float
    rank_ic: float
    sharpe: float
    max_drawdown: float
    turnover: float
    n_train: int
    n_test: int
    oos_returns: pd.Series = field(repr=False)


# ---------------------------------------------------------------------------
# Aggregate result container
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardResult:
    """Aggregated walk-forward validation result.

    Attributes
    ----------
    folds : list[FoldResult]
        Per-fold detail.
    mode : WindowMode
        Rolling or expanding.
    embargo_days : int
        Gap between train and test.
    deflated_sharpe : float | None
        Bailey & López de Prado (2014) deflated Sharpe ratio.  Populated
        after calling :meth:`compute_deflated_sharpe`.
    """

    folds: List[FoldResult]
    mode: WindowMode
    embargo_days: int
    deflated_sharpe: Optional[float] = None

    # -- convenience properties ---------------------------------------------

    @property
    def n_folds(self) -> int:
        return len(self.folds)

    @property
    def ics(self) -> np.ndarray:
        return np.array([f.ic for f in self.folds])

    @property
    def rank_ics(self) -> np.ndarray:
        return np.array([f.rank_ic for f in self.folds])

    @property
    def sharpes(self) -> np.ndarray:
        return np.array([f.sharpe for f in self.folds])

    @property
    def max_drawdowns(self) -> np.ndarray:
        return np.array([f.max_drawdown for f in self.folds])

    @property
    def turnovers(self) -> np.ndarray:
        return np.array([f.turnover for f in self.folds])

    @property
    def oos_returns(self) -> pd.Series:
        """Concatenated out-of-sample returns across all folds."""
        parts = [f.oos_returns for f in self.folds]
        return pd.concat(parts).sort_index()

    # -- aggregate stats with standard errors --------------------------------

    def summary(self, confidence: float = 0.95) -> pd.DataFrame:
        """Return a DataFrame of aggregate statistics with SEs and CIs.

        Parameters
        ----------
        confidence : float
            Confidence level for the interval (default 95 %).

        Returns
        -------
        pd.DataFrame
            Rows = metric, columns = [mean, std, se, ci_lower, ci_upper].
        """
        metrics: Dict[str, np.ndarray] = {
            "IC": self.ics,
            "Rank IC": self.rank_ics,
            "Sharpe": self.sharpes,
            "Max Drawdown": self.max_drawdowns,
            "Turnover": self.turnovers,
        }
        rows = []
        z = sp_stats.norm.ppf((1 + confidence) / 2)
        for name, vals in metrics.items():
            mu = np.nanmean(vals)
            sigma = np.nanstd(vals, ddof=1)
            se = sigma / np.sqrt(len(vals)) if len(vals) > 1 else np.nan
            rows.append(
                {
                    "metric": name,
                    "mean": mu,
                    "std": sigma,
                    "se": se,
                    "ci_lower": mu - z * se,
                    "ci_upper": mu + z * se,
                }
            )
        return pd.DataFrame(rows).set_index("metric")

    # -- Deflated Sharpe Ratio -----------------------------------------------

    def compute_deflated_sharpe(
        self,
        num_trials: int = 1,
        sharpe_benchmark: float = 0.0,
    ) -> float:
        r"""Compute the Deflated Sharpe Ratio (DSR).

        The DSR adjusts the observed Sharpe ratio for the number of trials
        (strategies tested), skewness, and kurtosis of returns, yielding a
        *p*-value-like probability that the observed Sharpe exceeds the
        benchmark purely by chance.

        See Bailey & López de Prado (2014) for derivation.

        Parameters
        ----------
        num_trials : int
            Total number of strategy variants tried (including this one).
        sharpe_benchmark : float
            Expected maximum Sharpe under the null.  When *num_trials* > 1
            the Euler–Mascheroni approximation is used automatically.

        Returns
        -------
        float
            DSR value in [0, 1].  Values > 0.95 indicate the observed
            Sharpe is unlikely to be due to chance alone.
        """
        oos = self.oos_returns.dropna()
        if len(oos) < 10:
            warnings.warn("Too few OOS observations for a reliable DSR.")
            self.deflated_sharpe = np.nan
            return np.nan

        T = len(oos)
        sr_obs = oos.mean() / oos.std() * np.sqrt(252)
        skew = float(sp_stats.skew(oos))
        kurt = float(sp_stats.kurtosis(oos, fisher=True))  # excess kurtosis

        # Expected max Sharpe under null (Euler-Mascheroni approx.)
        if num_trials > 1:
            euler_mascheroni = 0.5772156649
            sharpe_benchmark = np.sqrt(2 * np.log(num_trials)) - (
                np.log(np.pi) + euler_mascheroni
            ) / (2 * np.sqrt(2 * np.log(num_trials)))
            sharpe_benchmark *= np.sqrt(252) / np.sqrt(T)  # annualise

        # DSR test statistic
        # The variance of the Sharpe ratio estimator (Lo 2002, corrected
        # for non-normality) is:
        #   Var(SR) ≈ (1 - γ₃·SR + (γ₄-1)/4·SR²) / (T-1)
        # where γ₃ = skewness, γ₄ = excess kurtosis.
        sr_per_period = oos.mean() / oos.std()  # non-annualised
        var_sr = (
            1.0
            - skew * sr_per_period
            + ((kurt - 1) / 4.0) * sr_per_period ** 2
        ) / (T - 1)

        if var_sr <= 0:
            # Fallback: use simple SR variance = 1/(T-1)
            var_sr = 1.0 / (T - 1)

        sr_obs_pp = sr_per_period  # per-period observed SR
        sr_bench_pp = sharpe_benchmark / np.sqrt(252)  # de-annualise benchmark

        dsr = float(sp_stats.norm.cdf((sr_obs_pp - sr_bench_pp) / np.sqrt(var_sr)))
        self.deflated_sharpe = dsr
        return dsr

    # -- Plotting ------------------------------------------------------------

    def plot_cumulative(
        self,
        title: str = "Walk-Forward OOS Cumulative Returns",
        figsize: Tuple[int, int] = (14, 6),
        show_folds: bool = True,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot cumulative out-of-sample performance.

        Parameters
        ----------
        title : str
            Plot title.
        figsize : tuple
            Figure dimensions.
        show_folds : bool
            If *True*, shade alternate folds for visual clarity.
        save_path : str or None
            If given, save figure to this path.

        Returns
        -------
        matplotlib.figure.Figure
        """
        oos = self.oos_returns.sort_index()
        cum = (1 + oos).cumprod()

        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]})
        ax_cum, ax_dd = axes

        # -- Cumulative return line --
        ax_cum.plot(cum.index, cum.values, linewidth=1.2, color="#1f77b4")
        ax_cum.set_ylabel("Cumulative Return")
        ax_cum.set_title(title)
        ax_cum.axhline(1.0, color="grey", linewidth=0.6, linestyle="--")

        # -- Fold shading --
        if show_folds:
            for f in self.folds:
                color = "#e6f0ff" if f.fold_id % 2 == 0 else "#fff3e6"
                ax_cum.axvspan(f.test_start, f.test_end, alpha=0.25,
                               color=color, zorder=0)
                ax_dd.axvspan(f.test_start, f.test_end, alpha=0.25,
                              color=color, zorder=0)

        # -- Drawdown subplot --
        rolling_max = cum.cummax()
        drawdown = (cum - rolling_max) / rolling_max
        ax_dd.fill_between(drawdown.index, drawdown.values, 0,
                           color="#d62728", alpha=0.4)
        ax_dd.set_ylabel("Drawdown")
        ax_dd.set_xlabel("Date")

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Figure saved to %s", save_path)
        return fig


# ---------------------------------------------------------------------------
# Walk-Forward Validator
# ---------------------------------------------------------------------------

class WalkForwardValidator:
    """Rolling / expanding walk-forward cross-validator with embargo & purging.

    Parameters
    ----------
    train_window : int
        Number of calendar days in each training window (ignored in
        expanding mode after the first fold).
    test_window : int
        Number of calendar days in each test window.
    step_size : int
        Number of calendar days to advance between successive folds.
    min_train_size : int
        Minimum number of *observations* (not days) required in a training
        window.  Folds that do not meet this threshold are skipped.
    embargo_days : int
        Gap in calendar days between the end of the training set and the
        start of the test set to prevent information leakage from
        autocorrelated features.  Corresponds to *h* in López de Prado
        (2018, §7.4.1).
    mode : WindowMode or str
        ``"rolling"`` — fixed-length training window.
        ``"expanding"`` — training window grows with each fold.
    purge_days : int
        Additional days to remove from the end of the training set when
        target labels overlap with the test period (combinatorial purged
        CV concept from López de Prado 2018, §7.4.2).  Set to the forward
        return horizon used in label construction.

    Examples
    --------
    >>> validator = WalkForwardValidator(
    ...     train_window=504, test_window=63, step_size=21,
    ...     embargo_days=5, mode="rolling"
    ... )
    >>> for train_idx, test_idx in validator.split(dates):
    ...     model.fit(X[train_idx], y[train_idx])
    ...     preds = model.predict(X[test_idx])

    References
    ----------
    .. [1] López de Prado, M. (2018). Ch. 7.
    .. [2] Bailey, D. H. & López de Prado, M. (2014).
    """

    def __init__(
        self,
        train_window: int = 504,
        test_window: int = 63,
        step_size: int = 21,
        min_train_size: int = 100,
        embargo_days: int = 5,
        mode: Union[WindowMode, str] = WindowMode.ROLLING,
        purge_days: int = 0,
    ) -> None:
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        self.min_train_size = min_train_size
        self.embargo_days = embargo_days
        self.mode = WindowMode(mode)
        self.purge_days = purge_days

        # Validate
        if train_window <= 0 or test_window <= 0 or step_size <= 0:
            raise ValueError("train_window, test_window, step_size must be > 0")
        if embargo_days < 0 or purge_days < 0:
            raise ValueError("embargo_days and purge_days must be >= 0")

    # -- Core split logic ---------------------------------------------------

    def split(
        self,
        dates: Union[pd.DatetimeIndex, pd.Series, np.ndarray],
    ) -> Generator[IndexPair, None, None]:
        """Generate train/test index pairs honouring temporal ordering.

        Parameters
        ----------
        dates : array-like of datetime
            Sorted date array aligned with the rows of the feature matrix.

        Yields
        ------
        (train_indices, test_indices) : tuple[np.ndarray, np.ndarray]
            Integer positional indices into the original array.
        """
        dates = pd.DatetimeIndex(dates)
        if not dates.is_monotonic_increasing:
            raise ValueError("dates must be sorted in ascending order.")

        all_dates_unique = dates.unique().sort_values()
        start_date = all_dates_unique[0]
        end_date = all_dates_unique[-1]

        embargo_td = pd.Timedelta(days=self.embargo_days)
        purge_td = pd.Timedelta(days=self.purge_days)
        train_td = pd.Timedelta(days=self.train_window)
        test_td = pd.Timedelta(days=self.test_window)
        step_td = pd.Timedelta(days=self.step_size)

        # First fold: training ends at start + train_window
        fold_id = 0
        train_end_date = start_date + train_td

        while True:
            # Test window
            test_start_date = train_end_date + embargo_td + pd.Timedelta(days=1)
            test_end_date = test_start_date + test_td - pd.Timedelta(days=1)

            if test_start_date > end_date:
                break  # no room for a test window

            # Training window
            if self.mode == WindowMode.ROLLING:
                train_start_date = train_end_date - train_td
            else:  # expanding
                train_start_date = start_date

            # Apply purging: remove the last `purge_days` from training
            effective_train_end = train_end_date - purge_td

            # Build index masks
            train_mask = (dates >= train_start_date) & (dates <= effective_train_end)
            test_mask = (dates >= test_start_date) & (dates <= test_end_date)

            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]

            # Skip if insufficient data
            if len(train_idx) < self.min_train_size:
                logger.debug(
                    "Fold %d skipped: only %d train obs (min %d)",
                    fold_id, len(train_idx), self.min_train_size,
                )
                train_end_date += step_td
                continue

            if len(test_idx) == 0:
                logger.debug("Fold %d skipped: empty test set.", fold_id)
                train_end_date += step_td
                if train_end_date > end_date:
                    break
                continue

            yield train_idx, test_idx
            fold_id += 1
            train_end_date += step_td

            if train_end_date > end_date:
                break

    # -- High-level validate method -----------------------------------------

    def validate(
        self,
        alpha_func: Callable[[pd.DataFrame, pd.DataFrame], pd.Series],
        data: pd.DataFrame,
        returns: Optional[pd.Series] = None,
        date_col: str = "date",
        return_col: str = "forward_return",
        price_col: Optional[str] = "close",
    ) -> WalkForwardResult:
        """Run walk-forward validation on an alpha signal.

        Parameters
        ----------
        alpha_func : callable
            ``alpha_func(train_data, test_data) -> pd.Series``
            Must return a signal aligned with test_data index.  The function
            is trained on ``train_data`` and generates predictions on
            ``test_data``.
        data : pd.DataFrame
            Panel data containing features, dates, and forward returns.
        returns : pd.Series or None
            Pre-computed forward returns.  If *None*, uses ``return_col``
            from ``data``.
        date_col : str
            Column name for dates.
        return_col : str
            Column name for forward returns (used if ``returns`` is None).
        price_col : str or None
            Column name for price (used only for turnover calculation).

        Returns
        -------
        WalkForwardResult
        """
        if returns is None:
            if return_col not in data.columns:
                raise ValueError(
                    f"Column '{return_col}' not found and no `returns` supplied."
                )
            returns = data[return_col]

        dates = pd.DatetimeIndex(data[date_col])
        folds: List[FoldResult] = []

        for fold_id, (train_idx, test_idx) in enumerate(self.split(dates)):
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]
            test_returns = returns.iloc[test_idx]
            test_dates = dates[test_idx]

            # Generate signal
            try:
                signal = alpha_func(train_data, test_data)
            except Exception:
                logger.exception("alpha_func failed on fold %d", fold_id)
                continue

            # Align
            signal = signal.reindex(test_data.index)
            valid = signal.notna() & test_returns.notna()
            sig_valid = signal[valid]
            ret_valid = test_returns[valid]

            if len(sig_valid) < 5:
                logger.warning("Fold %d: <5 valid obs, skipping.", fold_id)
                continue

            # -- Per-fold metrics --
            ic = float(sig_valid.corr(ret_valid))
            rank_ic = float(sig_valid.rank().corr(ret_valid.rank()))

            # Signal-weighted daily returns
            weights = sig_valid / sig_valid.abs().sum()  # normalise
            oos_ret = (weights * ret_valid).groupby(test_dates[valid]).sum()
            sharpe = _annualised_sharpe(oos_ret)
            mdd = _max_drawdown(oos_ret)

            # Turnover
            turnover = _mean_turnover(weights)

            folds.append(
                FoldResult(
                    fold_id=fold_id,
                    train_start=pd.Timestamp(dates[train_idx[0]]),
                    train_end=pd.Timestamp(dates[train_idx[-1]]),
                    test_start=pd.Timestamp(test_dates[0]),
                    test_end=pd.Timestamp(test_dates[-1]),
                    ic=ic,
                    rank_ic=rank_ic,
                    sharpe=sharpe,
                    max_drawdown=mdd,
                    turnover=turnover,
                    n_train=len(train_idx),
                    n_test=len(test_idx),
                    oos_returns=oos_ret,
                )
            )
            logger.info(
                "Fold %d  IC=%.4f  RankIC=%.4f  Sharpe=%.2f  MDD=%.2f%%",
                fold_id, ic, rank_ic, sharpe, mdd * 100,
            )

        if not folds:
            raise RuntimeError(
                "No valid folds produced.  Check data length, window sizes, "
                "and min_train_size."
            )

        result = WalkForwardResult(
            folds=folds,
            mode=self.mode,
            embargo_days=self.embargo_days,
        )
        logger.info(
            "Walk-forward complete: %d folds, mean IC=%.4f, mean Sharpe=%.2f",
            result.n_folds,
            np.nanmean(result.ics),
            np.nanmean(result.sharpes),
        )
        return result


# ---------------------------------------------------------------------------
# Private helper functions
# ---------------------------------------------------------------------------

def _annualised_sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualised Sharpe ratio (zero risk-free rate)."""
    if returns.std() == 0 or len(returns) < 2:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))


def _max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown as a positive fraction."""
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max
    return float(-dd.min()) if len(dd) > 0 else 0.0


def _mean_turnover(weights: pd.Series) -> float:
    """Mean absolute change in weights (proxy for turnover)."""
    diff = weights.diff().abs()
    return float(diff.mean()) if len(diff) > 1 else 0.0
