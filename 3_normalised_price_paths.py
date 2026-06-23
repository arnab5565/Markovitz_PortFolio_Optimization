# ============================================================
#  FILE 3: Normalised Price Paths with ±1σ Rolling Bands
#  (Figure 1 from the report)
#  Indexes all assets to Base=100 on 1 Sep 2025 and plots
#  with ±1 standard deviation rolling bands.
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

# ── Normalise to Base = 100 at first date ────────────────────
df_norm = df / df.iloc[0] * 100

# ── Rolling window for bands (21 trading days ≈ 1 month) ────
window = 21
colors = plt.cm.tab10(np.linspace(0, 1, len(df_norm.columns)))

fig, ax = plt.subplots(figsize=(14, 7))

for i, col in enumerate(df_norm.columns):
    series      = df_norm[col]
    rolling_std = series.rolling(window).std()
    upper       = series + rolling_std
    lower       = series - rolling_std

    ax.plot(series.index, series, label=col,
            color=colors[i], linewidth=1.5)
    ax.fill_between(series.index, lower, upper,
                    color=colors[i], alpha=0.08)

ax.axhline(100, color="black", linestyle="--",
           linewidth=1, label="Base = 100")
ax.set_title("Figure 1: Normalised Price Paths (Base 100) with ±1σ Rolling Bands",
             fontsize=13, fontweight="bold")
ax.set_ylabel("Indexed Value (Base 100)")
ax.set_xlabel("Date")
ax.legend(loc="upper left", fontsize=8, ncol=2)
ax.grid(True, linestyle=":", alpha=0.5)

plt.tight_layout()
plt.savefig("normalised_price_paths.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: normalised_price_paths.png")
