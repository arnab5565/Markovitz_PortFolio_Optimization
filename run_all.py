# ============================================================
#  run_all.py  —  Execute all project scripts in order
#
#  Run this single file in Google Colab or locally to
#  reproduce the entire OR-II project from scratch:
#
#    !pip install yfinance pandas numpy scipy matplotlib seaborn
#    !python run_all.py
#
#  Or run step-by-step in Colab cells by copy-pasting each
#  numbered file individually.
# ============================================================

import subprocess
import sys

scripts = [
    ("1_data_download.py",         "Data Download & Preprocessing"),
    ("2_return_distributions.py",  "Return Distribution Histograms"),
    ("3_normalised_price_paths.py","Normalised Price Paths ±1σ"),
    ("4_monthly_return_heatmap.py","Monthly Return Heatmap"),
    ("5_maximum_drawdown.py",      "Maximum Drawdown Analysis"),
    ("6_covariance_matrix.py",     "Covariance & Correlation Matrix"),
    ("7_mvo_portfolio.py",         "MVO Portfolio Optimisation (10%)"),
    ("8_efficient_frontier.py",    "Efficient Frontier Plots"),
    ("9_comparison_AB.py",         "Comparison: 10% vs 15% Target"),
]

print("=" * 60)
print("  OR-II Project: MVO Portfolio  |  Sep 2025 → Mar 2026")
print("=" * 60)

for fname, desc in scripts:
    print(f"\n▶ Running {fname}  [{desc}] ...")
    result = subprocess.run(
        [sys.executable, fname],
        capture_output=False
    )
    if result.returncode != 0:
        print(f"  ✗ FAILED with return code {result.returncode}")
        break
    else:
        print(f"  ✓ Done")

print("\n" + "=" * 60)
print("  All scripts completed. Check output CSVs and PNGs.")
print("=" * 60)
