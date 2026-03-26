"""
run_first-5-alphas.py
──────────────────
Master runner: executes Alpha 01–05 sequentially and
generates the combined performance summary table.

Usage:
    python run_all_alphas.py                          # all alphas, equity
    python run_all_alphas.py --alphas 1 2 3           # specific alphas
    python run_all_alphas.py --alpha 4 --crypto       # alpha 4, crypto
    python run_all_alphas.py --start 2020-01-01       # custom date range
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
log = logging.getLogger("RunAll")

OUTPUT_DIR  = Path("./results")
REPORTS_DIR = OUTPUT_DIR / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def run_all(
    alphas:     list = None,
    start:      str  = "2015-01-01",
    end:        str  = "2024-12-31",
    use_crypto: bool = False,
) -> None:
    alphas = alphas or [1, 2, 3, 4, 5]

    results = []

    if 1 in alphas:
        log.info("══ Running Alpha 01 ══")
        from alpha_01_reversal_volume_decay import Alpha01
        a1 = Alpha01(start=start, end=end, use_crypto=use_crypto)
        a1.run(); a1.plot(); a1.generate_report()
        results.append(a1.metrics)

    if 2 in alphas:
        log.info("══ Running Alpha 02 ══")
        from alpha_02_vpin_filtered_momentum import Alpha02
        a2 = Alpha02(start=max(start, "2020-01-01"), end=end)
        a2.run(); a2.plot(); a2.generate_report()
        results.append(a2.metrics)

    if 3 in alphas:
        log.info("══ Running Alpha 03 ══")
        from alpha_03_amihud_illiquidity import Alpha03
        a3 = Alpha03(start=start, end=end, use_crypto=use_crypto)
        a3.run(); a3.plot(); a3.generate_report()
        results.append(a3.metrics)

    if 4 in alphas:
        log.info("══ Running Alpha 04 ══")
        from alpha_04_order_flow_imbalance import Alpha04
        a4 = Alpha04(start=max(start, "2022-01-01"), end=end)
        a4.run(); a4.plot(); a4.generate_report()
        results.append(a4.metrics)

    if 5 in alphas:
        log.info("══ Running Alpha 05 ══")
        from alpha_05_realized_skewness import Alpha05
        a5 = Alpha05(start=start, end=end, use_crypto=use_crypto)
        a5.run(); a5.plot(); a5.generate_report()
        results.append(a5.metrics)

    # ── Combined Summary Table ────────────────────────────────────────────────
    if results:
        summary = pd.DataFrame(results)
        csv_path = OUTPUT_DIR / "alpha_performance_summary.csv"
        summary.to_csv(csv_path, index=False)
        log.info("Summary saved → %s", csv_path)

        print("\n" + "=" * 80)
        print("COMBINED ALPHA PERFORMANCE SUMMARY")
        print("=" * 80)
        key_cols = [
            "alpha_id", "alpha_name",
            "IC_mean_IS_lag1", "IC_mean_OOS_lag1",
            "IC_mean_IS_lag22", "IC_mean_OOS_lag22",
            "Sharpe", "MaxDrawdown",
        ]
        display_cols = [c for c in key_cols if c in summary.columns]
        print(summary[display_cols].to_string(index=False))

        # README master table fragment
        md_rows = []
        for _, row in summary.iterrows():
            md_rows.append(
                f"| {row.get('alpha_id','—')} "
                f"| {row.get('alpha_name','—')} "
                f"| {row.get('IC_mean_OOS_lag1', row.get('IC_mean_OOS_lag5', row.get('IC_mean_OOS_lag1h', '—'))):.4f} "
                f"| {row.get('Sharpe', '—'):.2f} "
                f"| {row.get('MaxDrawdown', '—'):.2f} |"
            )
        md_table = "| # | Alpha | OOS IC | Sharpe | MaxDD |\n|---|---|---|---|---|\n" + "\n".join(md_rows)
        table_path = OUTPUT_DIR / "readme_table_fragment.md"
        table_path.write_text(md_table)
        log.info("README table fragment → %s", table_path)
        print("\nREADME table fragment:\n")
        print(md_table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all alphas")
    parser.add_argument("--alphas",  nargs="+", type=int, default=[1,2,3,4,5])
    parser.add_argument("--start",   default="2015-01-01")
    parser.add_argument("--end",     default="2024-12-31")
    parser.add_argument("--crypto",  action="store_true")
    args = parser.parse_args()

    run_all(
        alphas     = args.alphas,
        start      = args.start,
        end        = args.end,
        use_crypto = args.crypto,
    )
