"""
run_all_alphas.py
──────────────────
Master orchestrator: executes ALL 30 alphas sequentially, handles errors
per-alpha (continues on failure), and prints a summary report at the end.

Usage:
    python run_all_alphas.py                          # all 30 alphas
    python run_all_alphas.py --alphas 1 5 10 20       # specific alphas
    python run_all_alphas.py --start 2020-01-01       # custom date range
    python run_all_alphas.py --crypto                 # crypto mode where applicable
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("RunAll")

OUTPUT_DIR  = Path("./results")
REPORTS_DIR = OUTPUT_DIR / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Alpha registry: maps alpha number → (module_name, class_name, description)
# ─────────────────────────────────────────────────────────────────────────────

ALPHA_REGISTRY: dict[int, tuple[str, str, str]] = {
    1:  ("alpha_01_reversal_volume_decay",      "Alpha01", "Short-Term Reversal × Volume Decay"),
    2:  ("alpha_02_vpin_filtered_momentum",     "Alpha02", "VPIN-Filtered Momentum"),
    3:  ("alpha_03_amihud_illiquidity",         "Alpha03", "Amihud Illiquidity Factor"),
    4:  ("alpha_04_order_flow_imbalance",       "Alpha04", "Order Flow Imbalance"),
    5:  ("alpha_05_realized_skewness",          "Alpha05", "Realized Skewness"),
    6:  ("alpha_06_vol_term_structure",         "Alpha06", "Volatility Term Structure"),
    7:  ("alpha_07_cross_exchange_spread",      "Alpha07", "Cross-Exchange Spread (Crypto)"),
    8:  ("alpha_08_gbm_crypto_features",        "Alpha08", "GBM Ensemble + Crypto Features"),
    9:  ("alpha_09_hmm_regime_rotation",        "Alpha09", "HMM Regime × Factor Rotation"),
    10: ("alpha_10_kalman_dynamic_beta",        "Alpha10", "Kalman Dynamic Beta"),
    11: ("alpha_11_earnings_nlp_sentiment",     "Alpha11", "Earnings NLP Sentiment Drift"),
    12: ("alpha_12_google_trends_momentum",     "Alpha12", "Google Trends Attention Momentum"),
    13: ("alpha_13_cross_asset_macro",          "Alpha13", "Cross-Asset Macro Regime"),
    14: ("alpha_14_residual_momentum",          "Alpha14", "Residual Momentum"),
    15: ("alpha_15_onchain_supply_shock",       "Alpha15", "On-Chain Supply Shock (Crypto)"),
    16: ("alpha_16_funding_rate_carry",         "Alpha16", "Funding Rate Carry (Crypto)"),
    17: ("alpha_17_vol_skew_signal",            "Alpha17", "Options Vol Skew Signal"),
    18: ("alpha_18_variance_risk_premium",      "Alpha18", "Variance Risk Premium"),
    19: ("alpha_19_news_velocity",              "Alpha19", "News Velocity"),
    20: ("alpha_20_pcr_contrarian",             "Alpha20", "Put-Call Ratio Contrarian"),
    21: ("alpha_21_pead",                       "Alpha21", "Post-Earnings Announcement Drift"),
    22: ("alpha_22_eigenportfolio_statarb",     "Alpha22", "Eigenportfolio StatArb"),
    23: ("alpha_23_betting_against_beta",       "Alpha23", "Betting Against Beta"),
    24: ("alpha_24_quality_minus_junk",         "Alpha24", "Quality Minus Junk"),
    25: ("alpha_25_time_series_momentum",       "Alpha25", "Time-Series Momentum"),
    26: ("alpha_26_overnight_intraday",         "Alpha26", "Overnight-Intraday Decomposition"),
    27: ("alpha_27_dealer_gex",                 "Alpha27", "Dealer Gamma Exposure"),
    28: ("alpha_28_29_seasonality_squeeze",     "Alpha28", "Return Seasonality"),
    29: ("alpha_28_29_seasonality_squeeze",     "Alpha29", "Short Squeeze Detection"),
    30: ("alpha_30_index_reconstitution",       "Alpha30", "Index Reconstitution"),
}

ALL_ALPHA_IDS = sorted(ALPHA_REGISTRY.keys())


def run_single_alpha(
    alpha_id: int,
    start: str,
    end: str,
    use_crypto: bool,
) -> dict | None:
    """
    Import and run a single alpha. Returns its metrics dict or None on failure.
    """
    if alpha_id not in ALPHA_REGISTRY:
        log.warning("Alpha %02d not in registry — skipping", alpha_id)
        return None

    module_name, class_name, description = ALPHA_REGISTRY[alpha_id]
    log.info("══ Running Alpha %02d: %s ══", alpha_id, description)

    t0 = time.time()
    try:
        mod = importlib.import_module(module_name)
        cls = getattr(mod, class_name)

        # Some alphas require restricted date ranges or special flags
        kwargs: dict = {"start": start, "end": end}
        if use_crypto and alpha_id in {1, 3, 5, 7, 8, 15, 16}:
            kwargs["use_crypto"] = True
        if alpha_id == 2:
            kwargs["start"] = max(start, "2020-01-01")
        if alpha_id == 4:
            kwargs["start"] = max(start, "2022-01-01")

        alpha_instance = cls(**kwargs)
        alpha_instance.run()

        # Try to generate plots and reports (non-fatal if they fail)
        try:
            alpha_instance.plot()
        except Exception:
            log.debug("Alpha %02d: plot() not available or failed", alpha_id)

        try:
            alpha_instance.generate_report()
        except Exception:
            log.debug("Alpha %02d: generate_report() not available or failed", alpha_id)

        elapsed = time.time() - t0
        log.info("Alpha %02d completed in %.1fs", alpha_id, elapsed)

        metrics = getattr(alpha_instance, "metrics", None)
        if isinstance(metrics, dict):
            metrics["elapsed_seconds"] = round(elapsed, 1)
            return metrics
        return {"alpha_id": f"{alpha_id:02d}", "alpha_name": description, "status": "OK", "elapsed_seconds": round(elapsed, 1)}

    except Exception as exc:
        elapsed = time.time() - t0
        log.error("Alpha %02d FAILED after %.1fs: %s", alpha_id, elapsed, exc)
        log.debug(traceback.format_exc())
        return {
            "alpha_id": f"{alpha_id:02d}",
            "alpha_name": description,
            "status": "FAILED",
            "error": str(exc)[:200],
            "elapsed_seconds": round(elapsed, 1),
        }


def run_all(
    alphas: list[int] | None = None,
    start: str = "2015-01-01",
    end: str = "2024-12-31",
    use_crypto: bool = False,
) -> None:
    """Run all specified alphas and generate a summary report."""
    alphas = alphas or ALL_ALPHA_IDS
    total = len(alphas)

    log.info("=" * 70)
    log.info("ALPHA RESEARCH DISCOVERY — Running %d alpha(s)", total)
    log.info("Date range: %s → %s  |  Crypto mode: %s", start, end, use_crypto)
    log.info("=" * 70)

    results: list[dict] = []
    passed = 0
    failed = 0

    for i, alpha_id in enumerate(alphas, 1):
        log.info("[%d/%d] Alpha %02d", i, total, alpha_id)
        metrics = run_single_alpha(alpha_id, start, end, use_crypto)
        if metrics:
            results.append(metrics)
            if metrics.get("status") == "FAILED":
                failed += 1
            else:
                passed += 1

    # ── Summary Report ────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("ALPHA RESEARCH DISCOVERY — EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Total: {total}  |  Passed: {passed}  |  Failed: {failed}")
    print("-" * 80)

    if results:
        summary = pd.DataFrame(results)
        csv_path = OUTPUT_DIR / "alpha_performance_summary.csv"
        summary.to_csv(csv_path, index=False)
        log.info("Summary CSV saved → %s", csv_path)

        # Display key columns
        key_cols = [
            "alpha_id", "alpha_name", "status",
            "IC_mean_IS_lag1", "IC_mean_OOS_lag1",
            "IC_mean_IS_lag22", "IC_mean_OOS_lag22",
            "Sharpe", "MaxDrawdown", "elapsed_seconds",
        ]
        display_cols = [c for c in key_cols if c in summary.columns]
        print(summary[display_cols].to_string(index=False))

        # README master table fragment
        md_rows = []
        for _, row in summary.iterrows():
            status = row.get("status", "OK")
            if status == "FAILED":
                md_rows.append(
                    f"| {row.get('alpha_id', '—')} "
                    f"| {row.get('alpha_name', '—')} "
                    f"| FAILED | — | — |"
                )
            else:
                oos_ic = row.get("IC_mean_OOS_lag1", row.get("IC_mean_OOS_lag5", row.get("IC_mean_OOS_lag1h", "—")))
                sharpe = row.get("Sharpe", "—")
                maxdd = row.get("MaxDrawdown", "—")
                md_rows.append(
                    f"| {row.get('alpha_id', '—')} "
                    f"| {row.get('alpha_name', '—')} "
                    f"| {oos_ic if isinstance(oos_ic, str) else f'{oos_ic:.4f}'} "
                    f"| {sharpe if isinstance(sharpe, str) else f'{sharpe:.2f}'} "
                    f"| {maxdd if isinstance(maxdd, str) else f'{maxdd:.2f}'} |"
                )
        md_table = (
            "| # | Alpha | OOS IC | Sharpe | MaxDD |\n"
            "|---|---|---|---|---|\n"
            + "\n".join(md_rows)
        )
        table_path = OUTPUT_DIR / "readme_table_fragment.md"
        table_path.write_text(md_table)
        log.info("README table fragment → %s", table_path)

    # ── Failure details ───────────────────────────────────────────────────────
    if failed:
        print("\n" + "-" * 80)
        print("FAILURES:")
        for r in results:
            if r.get("status") == "FAILED":
                print(f"  Alpha {r['alpha_id']}: {r.get('error', 'unknown')}")
        print("-" * 80)

    print(f"\nResults directory: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run all 30 Alpha Research Discovery alphas"
    )
    parser.add_argument(
        "--alphas", nargs="+", type=int, default=None,
        help="Specific alpha numbers to run (default: all 30)",
    )
    parser.add_argument("--start",  default="2015-01-01", help="Backtest start date")
    parser.add_argument("--end",    default="2024-12-31", help="Backtest end date")
    parser.add_argument("--crypto", action="store_true",  help="Enable crypto mode")
    args = parser.parse_args()

    run_all(
        alphas     = args.alphas,
        start      = args.start,
        end        = args.end,
        use_crypto = args.crypto,
    )
