# ============================================================
#  FILE 6: Covariance Matrix & Annualised Returns (Section 4)
#  Computes the annualised covariance matrix from daily log-
#  returns and the sorted returns+std CSV needed by the MVO.
#
#  Annualisation:
#    • Return std dev  : daily_std × sqrt(2)  [6M → annual]
#    • Return          : 6M_return × 2        [6M → annual]
#    • Covariance      : daily_cov × 252      [trading days]
# ============================================================

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

tickers = [
    "MARUTI.NS", "EICHERMOT.NS", "ICICIBANK.NS",
    "SBIN.NS", "LT.NS", "BAJFINANCE.NS",
    "BHARTIARTL.NS", "RELIANCE.NS", "SUNPHARMA.NS",
    "GOLDBEES.NS"
]

# Desired display order (matching the report)
ORDER = [
    "RELIANCE.NS", "EICHERMOT.NS", "ICICIBANK.NS", "MARUTI.NS",
    "BHARTIARTL.NS", "BAJFINANCE.NS", "LT.NS",
    "SUNPHARMA.NS", "SBIN.NS", "GOLDBEES.NS"
]

# ── Download prices ──────────────────────────────────────────
df = yf.download(tickers, start="2025-09-01", end="2026-03-01",
                 auto_adjust=True)["Close"].ffill()

# ── Daily returns (simple %) ─────────────────────────────────
daily_returns = df.pct_change().dropna()

# ── 6-Month returns and std dev ──────────────────────────────
total_return = (df.iloc[-1] / df.iloc[0]) - 1          # simple 6M return
std_6m       = daily_returns.std() * np.sqrt(len(daily_returns))

# ── Annualise (×2 for return, ×sqrt(2) for std) ─────────────
ann_return   = total_return * 2
ann_std      = std_6m * np.sqrt(2)

# ── Build sorted returns CSV ─────────────────────────────────
result_df = pd.DataFrame({
    "Ticker":                     total_return.index,
    "Ticker_short":               [t.replace(".NS", "") for t in total_return.index],
    "6M Return (%)":              total_return.values * 100,
    "6M Std Dev (%)":             std_6m.values * 100,
    "Annualized Return (%)":      ann_return.values * 100,
    "Annualized Std Dev (%)":     ann_std.values * 100,
})
result_df = result_df.sort_values("6M Return (%)", ascending=False).reset_index(drop=True)
result_df.to_csv("6M_return_std_sorted.csv", index=False)
print("✅ Saved: 6M_return_std_sorted.csv")
print(result_df.to_string(index=False))

# ── Annualised Covariance Matrix (daily cov × 252) ──────────
cov_daily     = daily_returns.cov()
cov_annualized = cov_daily * 252

# Reorder rows/columns to match ORDER list
cov_annualized = cov_annualized.loc[ORDER, ORDER]
cov_annualized.index   = [t.replace(".NS", "") for t in cov_annualized.index]
cov_annualized.columns = [t.replace(".NS", "") for t in cov_annualized.columns]

cov_annualized.to_csv("annualized_covariance_matrix.csv")
print("\n✅ Saved: annualized_covariance_matrix.csv")
print("\nAnnualised Covariance Matrix:")
print(cov_annualized.round(6).to_string())

# ── Correlation Matrix (ordered) ─────────────────────────────
corr_matrix = daily_returns.corr()
corr_matrix = corr_matrix.loc[ORDER, ORDER]
corr_matrix.index   = [t.replace(".NS", "") for t in corr_matrix.index]
corr_matrix.columns = [t.replace(".NS", "") for t in corr_matrix.columns]
corr_matrix.to_csv("correlation_matrix_ordered.csv")
print("\n✅ Saved: correlation_matrix_ordered.csv")

# ── Plot Correlation Heatmap ─────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 10))
mask = np.zeros_like(corr_matrix, dtype=bool)   # show full matrix

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn",
    center=0,
    vmin=-1, vmax=1,
    linewidths=0.5,
    square=True,
    ax=ax,
    annot_kws={"size": 9}
)
ax.set_title("Portfolio Correlation Matrix (Ordered)",
             fontsize=14, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: correlation_heatmap.png")
