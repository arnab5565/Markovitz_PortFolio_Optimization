# ============================================================
#  FILE 2: Return Distribution Analysis (Section 3.1)
#  Plots histograms of daily log-returns with overlaid normal
#  curves. Reports skewness and excess kurtosis per asset.
# ============================================================

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

tickers = [
    "MARUTI.NS", "EICHERMOT.NS", "ICICIBANK.NS",
    "SBIN.NS", "LT.NS", "BAJFINANCE.NS",
    "BHARTIARTL.NS", "RELIANCE.NS", "SUNPHARMA.NS",
    "GOLDBEES.NS"
]

# ── Download & compute daily log-returns ─────────────────────
df = yf.download(tickers, start="2025-09-25", end="2026-03-24",
                 auto_adjust=True)["Close"].ffill()
log_returns = np.log(df / df.shift(1)).dropna()

# ── Plot: 5×2 grid of distribution histograms ───────────────
fig, axes = plt.subplots(5, 2, figsize=(16, 22))
axes = axes.flatten()
colors = plt.cm.tab10(np.linspace(0, 1, len(tickers)))
stats_results = []

for i, ticker in enumerate(log_returns.columns):
    name = ticker.replace(".NS", "")
    data = log_returns[ticker]

    # Histogram
    sns.histplot(data, kde=False, stat="density",
                 ax=axes[i], color=colors[i], alpha=0.5,
                 label=f"Actual: {name}")

    # Overlaid normal curve
    mu, std = data.mean(), data.std()
    x = np.linspace(data.min(), data.max(), 100)
    p = stats.norm.pdf(x, mu, std)
    axes[i].plot(x, p, "black", linewidth=2,
                 linestyle="--", label="Normal Curve")

    skew = stats.skew(data)
    kurt = stats.kurtosis(data)
    stats_results.append({
        "Company":         name,
        "Skewness":        round(skew, 4),
        "Excess Kurtosis": round(kurt, 4)
    })

    axes[i].set_title(f"{name} Distribution Analysis",
                      fontweight="bold", color=colors[i], fontsize=13)
    axes[i].legend(fontsize=8)
    axes[i].grid(axis="y", alpha=0.3)
    axes[i].set_xlabel(ticker)
    axes[i].set_ylabel("Density")

plt.tight_layout()
plt.savefig("return_distributions.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: return_distributions.png")

# ── Print normality stats ────────────────────────────────────
analysis_df = pd.DataFrame(stats_results).set_index("Company")
print("\n--- Statistical Deviations from Normality ---")
print(analysis_df.round(4))
analysis_df.to_csv("normality_stats.csv")
print("✅ Saved: normality_stats.csv")
