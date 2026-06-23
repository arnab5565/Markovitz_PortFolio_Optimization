# ============================================================
#  FILE 1: Data Download & Preprocessing
#  Downloads 6-Month price data (Sep 2025 → Mar 2026) from
#  yfinance, computes 6M returns and std dev, saves to CSV.
# ============================================================

import yfinance as yf
import pandas as pd
import numpy as np

tickers = [
    "MARUTI.NS", "EICHERMOT.NS", "ICICIBANK.NS",
    "SBIN.NS", "LT.NS", "BAJFINANCE.NS",
    "BHARTIARTL.NS", "RELIANCE.NS", "SUNPHARMA.NS",
    "GOLDBEES.NS"
]

start = "2025-09-01"
end   = "2026-03-01"

# ── Download closing prices ──────────────────────────────────
data = yf.download(tickers, start=start, end=end)["Close"]
data = data.fillna(method="ffill")

# ── 6-Month simple return: (P_end / P_start) - 1 ───────────
total_return = (data.iloc[-1] / data.iloc[0]) - 1

# ── 6-Month std dev: daily std × sqrt(T trading days) ───────
returns      = data.pct_change().dropna()
std_6m       = returns.std() * np.sqrt(len(returns))

# ── Build result DataFrame ───────────────────────────────────
result = pd.DataFrame({
    "Ticker":          total_return.index,
    "6M Return (%)":   total_return.values * 100,
    "6M Std Dev (%)":  std_6m.values * 100
})

result = result.sort_values(by="6M Return (%)", ascending=False).reset_index(drop=True)

print("=" * 55)
print("  6-Month Return & Std Dev  |  Sep 2025 → Mar 2026")
print("=" * 55)
print(result.to_string(index=False))

# ── Save ─────────────────────────────────────────────────────
result.to_csv("6M_return_std.csv", index=False)
print("\n✅ Saved: 6M_return_std.csv")
