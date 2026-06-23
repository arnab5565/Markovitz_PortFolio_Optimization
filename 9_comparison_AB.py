# ============================================================
#  FILE 9: Side-by-Side Comparison — Problem A (10%) vs B (15%)
#  (Section 7.5 of the report)
#  Runs BOTH optimisations and prints the comparison table
#  exactly as shown in the report, plus saves to CSV.
# ============================================================

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ── Load Data ────────────────────────────────────────────────
returns_df = pd.read_csv("6M_return_std_sorted.csv")
cov_df     = pd.read_csv("annualized_covariance_matrix.csv", index_col=0)

returns_df["Ticker_short"] = returns_df["Ticker"].str.replace(".NS", "", regex=False)
returns_df["Annualized Return (%)"] = returns_df["6M Return (%)"] * 2

tickers    = returns_df["Ticker_short"].tolist()
mu         = returns_df["Annualized Return (%)"].values / 100
sigma_i    = returns_df["6M Std Dev (%)"].values          # 6M std dev %
cov_matrix = cov_df.loc[tickers, tickers].values

n  = len(tickers)
rf = 0.06

# ── MVO Solver ───────────────────────────────────────────────
def solve_mvo(target_ret, upper_bound=0.40):
    constraints = [
        {"type": "eq",   "fun": lambda w: np.sum(w) - 1},
        {"type": "ineq", "fun": lambda w: np.dot(w, mu) - target_ret},
    ]
    bounds = [(0.0, upper_bound)] * n
    res = minimize(
        lambda w: w @ cov_matrix @ w,
        np.ones(n) / n,
        jac=lambda w: 2 * cov_matrix @ w,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    if not res.success:
        raise RuntimeError(f"Optimisation failed: {res.message}")
    w = res.x
    w[w < 1e-4] = 0.0
    w /= w.sum()
    return w

wA = solve_mvo(0.10)
wB = solve_mvo(0.15)

# ── Portfolio metrics ────────────────────────────────────────
def portfolio_metrics(w):
    ret = float(w @ mu)
    var = float(w @ cov_matrix @ w)
    std = float(np.sqrt(var))
    sr  = (ret - rf) / std
    return ret, var, std, sr

rA, vA, sA, srA = portfolio_metrics(wA)
rB, vB, sB, srB = portfolio_metrics(wB)

# ── Per-asset results for both targets ───────────────────────
df_comp = pd.DataFrame({
    "Asset":        returns_df["Ticker"].tolist(),
    "r_i (%)":      np.round(mu * 100, 5),
    "σ_i (%)":      np.round(sigma_i, 3),
    "Wt A (%)":     np.round(wA * 100, 4),
    "Wt B (%)":     np.round(wB * 100, 4),
    "ΔWeight":      np.round((wB - wA) * 100, 4),
    "Ret A":        np.round(wA * mu * 100, 4),
    "Ret B":        np.round(wB * mu * 100, 4),
})

SEP = "=" * 80

# ── Print comparison ─────────────────────────────────────────
print(SEP)
print("  SIDE-BY-SIDE COMPARISON: Problem A (10%) vs Problem B (15%)")
print(SEP)
pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.width", 120)
print(df_comp.to_string(index=False))

print(f"\n{SEP}")
print(f"  {'Metric':<35} {'Problem A (10%)':>18} {'Problem B (15%)':>18}")
print(SEP)
print(f"  {'Return Achieved':<35} {'10.00%':>18} {'15.00%':>18}")
print(f"  {'Portfolio Std Dev':<35} {sA*100:>17.2f}% {sB*100:>17.2f}%")
print(f"  {'Portfolio Variance':<35} {vA:>18.5f} {vB:>18.5f}")
print(f"  {'Sharpe Ratio':<35} {srA:>18.4f} {srB:>18.4f}")
print(f"  {'GOLDBEES Standalone Std Dev':<35} {'36.57% (Ann.)':>18} {'36.57% (Ann.)':>18}")
print(f"  {'Risk Reduction vs Gold':<35} {36.57 - sA*100:>17.2f}% {36.57 - sB*100:>17.2f}%")
best_eq_A  = returns_df.loc[wA.argmax(), "Ticker"]
worst_eq_A = returns_df.loc[wA[:-1].argmin(), "Ticker"]   # exclude GOLDBEES
print(f"  {'Best Equity Weight':<35} {best_eq_A+f' {wA.max()*100:.2f}%':>18}"
      f" {returns_df.loc[wB.argmax(),'Ticker']+f' {wB.max()*100:.2f}%':>18}")
print(SEP)

# ── Save ─────────────────────────────────────────────────────
df_comp.to_csv("comparison_A_vs_B.csv", index=False)
print("\n✅ Saved: comparison_A_vs_B.csv")

summary_both = pd.DataFrame({
    "Metric":    ["Return (%)", "Std Dev (%)", "Variance",
                  "Sharpe Ratio", "Active Assets"],
    "Problem A": [rA*100, sA*100, vA, srA, int((wA > 0).sum())],
    "Problem B": [rB*100, sB*100, vB, srB, int((wB > 0).sum())],
})
summary_both.to_csv("summary_both_targets.csv", index=False)
print("✅ Saved: summary_both_targets.csv")
