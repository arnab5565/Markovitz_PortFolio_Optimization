# ============================================================
#  FILE 4: Monthly Return Heatmap  (Figure 2 from the report)
#  Shows month-by-month % returns per asset as a colour-coded
#  heatmap — deep red = heavy losses, green = gains.
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

# ── Download & compute monthly returns ───────────────────────
df = yf.download(tickers, start="2025-09-25", end="2026-03-24",
                 auto_adjust=True)["Close"].ffill()
df.columns = [c.replace(".NS", "") for c in df.columns]

monthly_prices  = df.resample("ME").last()
monthly_returns = monthly_prices.pct_change().dropna() * 100   # in %

# Format column labels as "Oct-2025", "Nov-2025" etc.
monthly_returns.index = monthly_returns.index.strftime("%b-%Y")
monthly_returns = monthly_returns.T   # companies as rows, months as cols

# ── Plot heatmap ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 8))

sns.heatmap(
    monthly_returns,
    annot=True,
    fmt=".1f",
    cmap="RdYlGn",
    center=0,
    linewidths=0.5,
    linecolor="white",
    cbar_kws={"label": "Monthly Return (%)"},
    ax=ax
)

ax.set_title("Figure 2: Monthly Return Heatmap (%)",
             fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Month-Year", fontsize=11)
ax.set_ylabel("Company", fontsize=11)
ax.tick_params(axis="x", rotation=0)
ax.tick_params(axis="y", rotation=0)

plt.tight_layout()
plt.savefig("monthly_return_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: monthly_return_heatmap.png")
