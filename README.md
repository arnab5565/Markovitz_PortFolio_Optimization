# OR-II Project: Mean-Variance Portfolio Optimization
**Period:** Sep 2025 → Mar 2026 | **Assets:** 9 Nifty50 stocks + GOLDBEES  
**By:** Vivek Dubey (24IM10070), Arnab Maiti (24IM10017), Kartik Jeengar (24IM10010)

---

## Setup (run once in Colab)
```
!pip install yfinance pandas numpy scipy matplotlib seaborn
```

## File Map

| File | Section in Report | What it does | Outputs |
|------|------------------|--------------|---------|
| `1_data_download.py` | 2.2 | Downloads prices, computes 6M return & std dev | `6M_return_std.csv` |
| `2_return_distributions.py` | 3.1 | Histograms with normal curves, skewness & kurtosis | `return_distributions.png`, `normality_stats.csv` |
| `3_normalised_price_paths.py` | Fig 1 | Price paths indexed to Base=100, ±1σ rolling bands | `normalised_price_paths.png` |
| `4_monthly_return_heatmap.py` | Fig 2 | Monthly return heatmap | `monthly_return_heatmap.png` |
| `5_maximum_drawdown.py` | 3.4 | Drawdown curves, MDD summary table | `maximum_drawdown.png`, `max_drawdown_summary.csv` |
| `6_covariance_matrix.py` | 4 | Annualised cov & corr matrices, sorted returns CSV | `annualized_covariance_matrix.csv`, `6M_return_std_sorted.csv`, `correlation_heatmap.png` |
| `7_mvo_portfolio.py` | 6 | **Core MVO optimizer** (SLSQP, target=10%) | `mvo_allocation_results.csv`, `mvo_portfolio_summary.csv` |
| `8_efficient_frontier.py` | 7 | Efficient frontier, CML, allocation pies & risk bars | `efficient_frontier.png`, `portfolio_allocation.png` |
| `9_comparison_AB.py` | 7.5 | Both targets (10% & 15%) side-by-side comparison | `comparison_A_vs_B.csv`, `summary_both_targets.csv` |
| `run_all.py` | — | Runs all scripts in order | all of the above |

## Run Order
```
# Option 1: All at once
python run_all.py

# Option 2: Step by step (recommended in Colab)
python 1_data_download.py
python 2_return_distributions.py
...
python 9_comparison_AB.py
```

## Key Parameters (edit in each file)
| Parameter | Default | Location |
|-----------|---------|----------|
| `start` / `end` | 2025-09-01 / 2026-03-01 | Files 1,3,4,5,6 |
| `target_ret` | 0.10 (10%) | File 7 |
| `rf` | 0.06 (6%) | Files 7,8,9 |
| `upper_bound` | 0.40 (40%) | Files 7,8,9 |
| `TARGET_B` | 0.15 (15%) | Files 8,9 |

## Model Formulation
```
min  x^T Σ x
s.t. Σ r_i x_i ≥ T      [C1: return constraint]
     Σ x_i = 1           [C2: budget constraint]
     x_i ≥ 0             [C3: non-negativity]
     x_i ≤ 0.40          [C4: upper bound per asset]
```
