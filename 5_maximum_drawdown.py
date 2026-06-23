# ============================================================
#  FILE 5: Maximum Drawdown Analysis  (Section 3.4)
#  Drawdown(t) = (P_t - max(P_1..t)) / max(P_1..t) × 100
#  Plots per-asset drawdown curves and prints MDD summary.
# ============================================================

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

tickers = [
    "MARUTI.NS", "EICHERMOT.NS", "ICICIBANK.NS",
    "SBIN.NS", "LT.NS", "BAJFINANCE.NS",
    "BHARTIARTL.NS", "RELIANCE.NS", "SUNPHARMA.NS",
    "GOLDBEES.NS"
]

# ── Download prices ──────────────────────────────────────────
df = yf.download(tickers, start="2025-09-25", end="2026-03-24",
                 auto_adjust=True)["Close"].ffill()
df.columns = [c.replace(".NS", "") for c in df.columns]

# ── Compute drawdown series ──────────────────────────────────
rolling_max = df.cummax()
drawdowns   = (df - rolling_max) / rolling_max * 100   # in %

# ── Plot 5×2 grid ────────────────────────────────────────────
fig, axes = plt.subplots(5, 2, figsize=(16, 22))
axes = axes.flatten()
colors = plt.cm.tab10(np.linspace(0, 1, len(df.columns)))

for i, col in enumerate(drawdowns.columns):
    mdd_value = drawdowns[col].min()

    axes[i].plot(drawdowns.index, drawdowns[col],
                 color=colors[i], lw=2)
    axes[i].fill_between(drawdowns.index, drawdowns[col], 0,
                         color=colors[i], alpha=0.2)

    axes[i].set_title(
        f"{col} Drawdown (Max: {mdd_value:.2f}%)",
        fontweight="bold", fontsize=14, color=colors[i]
    )
    axes[i].set_ylabel("Loss from Peak (%)")
    axes[i].axhline(0, color="black", lw=1, linestyle="--")
    axes[i].grid(True, linestyle=":", alpha=0.6)
    axes[i].set_ylim(drawdowns.min().min() - 2, 1)

plt.tight_layout()
plt.savefig("maximum_drawdown.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: maximum_drawdown.png")

# ── Summary table ────────────────────────────────────────────
mdd_table = (
    drawdowns.min()
    .to_frame(name="Max Drawdown (%)")
    .sort_values("Max Drawdown (%)")
)
print("\n--- Summary: Maximum Drawdown by Company ---")
print(mdd_table.round(2))
mdd_table.to_csv("max_drawdown_summary.csv")
print("✅ Saved: max_drawdown_summary.csv")
