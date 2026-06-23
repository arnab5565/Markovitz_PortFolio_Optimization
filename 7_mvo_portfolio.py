# ============================================================
#  FILE 7: mvo_portfolio.py  (Section 6 — Algorithmic Execution)
#  Full Markowitz Mean-Variance Optimization using SciPy SLSQP.
#
#  Inputs  : 6M_return_std_sorted.csv
#             annualized_covariance_matrix.csv
#  Outputs : mvo_allocation_results.csv
#             mvo_portfolio_summary.csv
#
#  Model:
#    min  x^T Σ x
#    s.t. Σ r_i x_i ≥ T   (return constraint)
#         Σ x_i = 1        (budget constraint)
#         x_i ≥ 0          (non-negativity)
#         x_i ≤ 0.40       (upper bound per asset)
# ============================================================

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ── 1. Load Data ─────────────────────────────────────────────
returns_df = pd.read_csv("6M_return_std_sorted.csv")
cov_df     = pd.read_csv("annualized_covariance_matrix.csv", index_col=0)

# Strip ".NS" suffix so tickers match covariance matrix index
returns_df["Ticker_short"] = returns_df["Ticker"].str.replace(".NS", "", regex=False)

# Annualise 6M returns (×2)
returns_df["Annualized Return (%)"] = returns_df["6M Return (%)"] * 2

tickers    = returns_df["Ticker_short"].tolist()
mu         = returns_df["Annualized Return (%)"].values / 100   # as decimals
cov_matrix = cov_df.loc[tickers, tickers].values

n          = len(tickers)
rf         = 0.06          # risk-free rate (annualised)
target_ret = 0.10          # 10% annualised target return

# ── 2. MVO Optimisation ──────────────────────────────────────

def port_variance(w):
    return w @ cov_matrix @ w

def port_variance_grad(w):
    return 2 * cov_matrix @ w

constraints = [
    {"type": "eq",   "fun": lambda w: np.sum(w) - 1},            # C2: budget
    {"type": "ineq", "fun": lambda w: np.dot(w, mu) - target_ret}, # C1: return
]
bounds = [(0.0, 0.40)] * n      # C3 + C4: non-neg & upper bound

w0 = np.ones(n) / n             # equal-weight starting point

result = minimize(
    port_variance,
    w0,
    jac=port_variance_grad,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"ftol": 1e-12, "maxiter": 2000},
)

if not result.success:
    raise RuntimeError(f"Optimisation failed: {result.message}")

weights = result.x
# Zero out negligible weights (< 0.01%)
weights[weights < 1e-4] = 0.0
weights /= weights.sum()    # renormalise after zeroing

# ── 3. Portfolio-Level Metrics ───────────────────────────────
port_return       = float(weights @ mu)
port_variance_val = float(weights @ cov_matrix @ weights)
port_std          = float(np.sqrt(port_variance_val))
sharpe_ratio      = (port_return - rf) / port_std

# ── 4. Return Contributions ──────────────────────────────────
# Return contribution of asset i = w_i × μ_i
return_contributions  = weights * mu                                         # absolute (decimal)
return_contrib_pct    = return_contributions / port_return * 100             # % of total return

# ── 5. Risk Contributions (Marginal / Component) ─────────────
# Marginal contribution to risk = (Σw)_i / σ_p
# Component risk contribution   = w_i × MCR_i
# % risk contribution           = CRC_i / σ_p × 100
marginal_risk     = (cov_matrix @ weights) / port_std          # MCR vector
component_risk    = weights * marginal_risk                    # CRC vector
risk_contrib_pct  = component_risk / port_std * 100            # % of total variance

# ── 6. Results Table ─────────────────────────────────────────
results = pd.DataFrame({
    "Ticker":                    returns_df["Ticker"].tolist(),
    "Ann. Return (%)":           np.round(mu * 100, 4),
    "Weight (%)":                np.round(weights * 100, 4),
    "Return Contribution (%)":   np.round(return_contributions * 100, 4),
    "% of Port. Return":         np.round(return_contrib_pct, 4),
    "Marginal Risk Contrib":     np.round(marginal_risk, 6),
    "Component Risk Contrib":    np.round(component_risk, 6),
    "% of Port. Risk":           np.round(risk_contrib_pct, 4),
})

# ── 7. Print Output ──────────────────────────────────────────
SEP = "=" * 80

print(SEP)
print("  MEAN-VARIANCE OPTIMISATION  |  "
      f"Target Return: {target_ret*100:.0f}%  |  RF: {rf*100:.0f}%")
print(SEP)

print("\n ASSET-LEVEL BREAKDOWN\n")
pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.max_columns", 10)
pd.set_option("display.width", 120)
print(results.to_string(index=False))

print(f"\n{SEP}")
print("  PORTFOLIO SUMMARY")
print(SEP)
print(f"  {'Portfolio Expected Return:':<35} {port_return*100:>10.4f} %")
print(f"  {'Portfolio Variance:':<35} {port_variance_val:>10.6f}")
print(f"  {'Portfolio Std Deviation:':<35} {port_std*100:>10.4f} %")
print(f"  {'Sharpe Ratio (RF = 6%):':<35} {sharpe_ratio:>10.4f}")
print(SEP)

print("\n WEIGHT ALLOCATION SUMMARY\n")
active = results[results["Weight (%)"] > 0].copy()
for _, row in active.iterrows():
    bar = "█" * int(row["Weight (%)"] / 2)
    print(f"  {row['Ticker']:<15} {row['Weight (%)']:>6.2f}%  {bar}")

print(f"\n  {'TOTAL':<15} {active['Weight (%)'].sum():>6.2f}%")
print(f"\n  Stocks with zero allocation: "
      f"{', '.join(results[results['Weight (%)']==0]['Ticker'].tolist()) or 'None'}")

# ── 8. Save to CSV ───────────────────────────────────────────
results.to_csv("mvo_allocation_results.csv", index=False)

summary = pd.DataFrame({
    "Metric": [
        "Target Return (%)",
        "Portfolio Expected Return (%)",
        "Portfolio Variance",
        "Portfolio Std Deviation (%)",
        "Sharpe Ratio",
        "Risk-Free Rate (%)",
    ],
    "Value": [
        target_ret * 100,
        round(port_return * 100, 4),
        round(port_variance_val, 6),
        round(port_std * 100, 4),
        round(sharpe_ratio, 4),
        rf * 100,
    ]
})
summary.to_csv("mvo_portfolio_summary.csv", index=False)

print("\n  Results saved to:")
print("     • mvo_allocation_results.csv")
print("     • mvo_portfolio_summary.csv")
print(SEP)
