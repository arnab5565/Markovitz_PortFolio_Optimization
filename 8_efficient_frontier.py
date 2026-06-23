# ============================================================
#  FILE 8: Efficient Frontier + Portfolio Allocation Visuals
#  (Section 7 — Performance Metrics & Frontier Analysis)
#
#  Plots:
#    1. Efficient Frontier with CML, MVP, Max Sharpe,
#       10% target and 15% target portfolio points
#    2. Side-by-side pie charts (weight allocation)
#    3. Side-by-side bar charts (risk contribution)
#
#  Run AFTER 7_mvo_portfolio.py so CSVs exist.
#  Or set RUN_INLINE = True to recompute internally.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize

# ── Configuration ────────────────────────────────────────────
RF         = 0.06
TARGET_A   = 0.10     # Problem A
TARGET_B   = 0.15     # Problem B
N_FRONTIER = 300      # points on the frontier

# ── Load pre-computed data ───────────────────────────────────
returns_df = pd.read_csv("6M_return_std_sorted.csv")
cov_df     = pd.read_csv("annualized_covariance_matrix.csv", index_col=0)

returns_df["Ticker_short"] = returns_df["Ticker"].str.replace(".NS", "", regex=False)
returns_df["Annualized Return (%)"] = returns_df["6M Return (%)"] * 2

tickers    = returns_df["Ticker_short"].tolist()
mu         = returns_df["Annualized Return (%)"].values / 100
cov_matrix = cov_df.loc[tickers, tickers].values
n          = len(tickers)

# ── Helper: solve MVO for a given target ────────────────────
def solve_mvo(target, upper_bound=0.40):
    constraints = [
        {"type": "eq",   "fun": lambda w: np.sum(w) - 1},
        {"type": "ineq", "fun": lambda w: np.dot(w, mu) - target},
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
    if res.success:
        w = res.x
        w[w < 1e-4] = 0.0
        w /= w.sum()
        std = np.sqrt(w @ cov_matrix @ w)
        ret = w @ mu
        return w, ret, std
    return None, None, None

# ── Build Efficient Frontier ─────────────────────────────────
ret_min  = max(mu.min(), RF + 0.001)
ret_max  = mu.max() * 0.95
targets  = np.linspace(ret_min, ret_max, N_FRONTIER)

frontier_std  = []
frontier_ret  = []
for t in targets:
    _, r, s = solve_mvo(t)
    if r is not None:
        frontier_std.append(s * 100)
        frontier_ret.append(r * 100)

frontier_std = np.array(frontier_std)
frontier_ret = np.array(frontier_ret)

# ── Minimum Variance Portfolio ───────────────────────────────
mvp_idx      = np.argmin(frontier_std)
mvp_std      = frontier_std[mvp_idx]
mvp_ret      = frontier_ret[mvp_idx]

# ── Max Sharpe Portfolio ─────────────────────────────────────
sharpes      = (frontier_ret/100 - RF) / (frontier_std/100)
ms_idx       = np.argmax(sharpes)
ms_std       = frontier_std[ms_idx]
ms_ret       = frontier_ret[ms_idx]
ms_sr        = sharpes[ms_idx]

# ── Solve for both targets ───────────────────────────────────
wA, rA, sA = solve_mvo(TARGET_A)
wB, rB, sB = solve_mvo(TARGET_B)
srA = (rA - RF) / sA
srB = (rB - RF) / sB

# Individual asset points
asset_stds = np.sqrt(np.diag(cov_matrix)) * 100
asset_rets = mu * 100

# ── CML line ─────────────────────────────────────────────────
cml_x = np.linspace(0, frontier_std.max() * 1.1, 200)
cml_y = RF * 100 + (ms_ret - RF * 100) / ms_std * cml_x

# ═══════════════════════════════════════════════════════════
#  FIGURE 1: Efficient Frontier
# ═══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(13, 9))

# Inefficient part (below MVP)
ax.plot(frontier_std[:mvp_idx+1], frontier_ret[:mvp_idx+1],
        "gray", lw=1.5, linestyle="--", label="Inefficient Frontier")

# Efficient part (above MVP)
ax.plot(frontier_std[mvp_idx:], frontier_ret[mvp_idx:],
        "cyan", lw=2.5, label="Efficient Frontier")

# CML
ax.plot(cml_x, cml_y, "yellow", lw=1.5, linestyle="-.",
        label="Capital Market Line (CML)")

# Individual assets
ax.scatter(asset_stds, asset_rets, c="white", s=80,
           edgecolors="gray", zorder=5, label="Individual Assets")
for i, t in enumerate(tickers):
    ax.annotate(t, (asset_stds[i], asset_rets[i]),
                fontsize=7, color="white",
                textcoords="offset points", xytext=(5, 3))

# MVP
ax.scatter(mvp_std, mvp_ret, color="lime", s=150, zorder=6,
           label=f"Min Variance σ={mvp_std:.2f}% r={mvp_ret:.2f}%")

# Max Sharpe
ax.scatter(ms_std, ms_ret, color="gold", s=200, marker="*", zorder=7,
           label=f"Max Sharpe SR={ms_sr:.3f}")
ax.annotate("Max Sharpe", (ms_std, ms_ret), color="gold",
            fontsize=9, textcoords="offset points", xytext=(8, -12))

# 10% Target
ax.scatter(sA * 100, rA * 100, color="orange", s=200, marker="D", zorder=7,
           label=f"10% Target σ={sA*100:.2f}% SR={srA:.3f}")
ax.annotate(f"10% Target\nSR={srA:.3f}", (sA*100, rA*100),
            color="orange", fontsize=8,
            textcoords="offset points", xytext=(-60, -30))

# 15% Target
ax.scatter(sB * 100, rB * 100, color="red", s=200, marker="D", zorder=7,
           label=f"15% Target σ={sB*100:.2f}% SR={srB:.3f}")
ax.annotate(f"15% Target\nSR={srB:.3f}", (sB*100, rB*100),
            color="red", fontsize=8,
            textcoords="offset points", xytext=(-20, -35))

# RF point
ax.scatter(0, RF * 100, color="white", s=100, marker="^", zorder=6,
           label=f"RF = {RF*100:.0f}%")

ax.set_facecolor("#1a1a2e")
fig.patch.set_facecolor("#1a1a2e")
ax.tick_params(colors="white")
ax.xaxis.label.set_color("white")
ax.yaxis.label.set_color("white")
ax.title.set_color("white")
for spine in ax.spines.values():
    spine.set_edgecolor("gray")

ax.set_title(
    "Efficient Frontier — MVO Portfolio\n"
    "10 Assets  |  Long-Only  |  RF = 6%  |  6M Data (Annualized)",
    fontsize=13, fontweight="bold"
)
ax.set_xlabel("Portfolio Standard Deviation (%)", fontsize=11)
ax.set_ylabel("Portfolio Expected Return (%)", fontsize=11)
ax.legend(loc="upper left", fontsize=8,
          facecolor="#1a1a2e", labelcolor="white", edgecolor="gray")
ax.grid(True, linestyle=":", alpha=0.3, color="gray")

plt.tight_layout()
plt.savefig("efficient_frontier.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.show()
print("✅ Saved: efficient_frontier.png")

# ═══════════════════════════════════════════════════════════
#  FIGURE 2: Portfolio Allocation — Pie + Risk Bar (A & B)
# ═══════════════════════════════════════════════════════════
labels = returns_df["Ticker"].tolist()
short  = [t.replace(".NS", "") for t in labels]

# Risk contribution % for A and B
def risk_contrib_pct(w, cov, std):
    mcr = (cov @ w) / std
    crc = w * mcr
    return crc / std * 100

rcA = risk_contrib_pct(wA, cov_matrix, sA)
rcB = risk_contrib_pct(wB, cov_matrix, sB)

colors_pie = plt.cm.tab10(np.linspace(0, 1, n))

fig = plt.figure(figsize=(18, 14))
fig.suptitle("MVO Portfolio Analysis — Target Return: 10% & 15%",
             fontsize=15, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

for row, (w, rc, target, sr, std) in enumerate([
        (wA, rcA, "10%", srA, sA),
        (wB, rcB, "15%", srB, sB)
]):
    active_mask = w > 0.001

    # ── Pie chart ─────────────────────────────────────────
    ax_pie = fig.add_subplot(gs[row, 0])
    wedge_vals   = w[active_mask] * 100
    wedge_labels = np.array(short)[active_mask]
    wedge_colors = colors_pie[active_mask]

    ax_pie.pie(wedge_vals, labels=wedge_labels,
               autopct="%1.1f%%", startangle=140,
               colors=wedge_colors, pctdistance=0.82,
               textprops={"fontsize": 8})
    ax_pie.set_title(
        f"{target} | Weight Allocation (%)\n"
        f"Return: {w@mu*100:.2f}%  Std Dev: {std*100:.2f}%  "
        f"Sharpe: {sr:.3f}",
        fontsize=10, fontweight="bold"
    )

    # ── Horizontal bar chart (risk contribution) ──────────
    ax_bar = fig.add_subplot(gs[row, 1])
    bar_vals   = rc
    bar_colors = [colors_pie[i] if bar_vals[i] > 0 else "gray"
                  for i in range(n)]
    bars = ax_bar.barh(short, bar_vals, color=bar_colors, edgecolor="white")

    for bar, val in zip(bars, bar_vals):
        if abs(val) > 0.3:
            ax_bar.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                        f"{val:.1f}%", va="center", fontsize=8)

    ax_bar.set_title(f"{target} | Risk Contribution (%)",
                     fontsize=10, fontweight="bold")
    ax_bar.set_xlabel("% of Portfolio Risk")
    ax_bar.axvline(0, color="black", lw=0.8)
    ax_bar.grid(True, axis="x", linestyle=":", alpha=0.5)

plt.savefig("portfolio_allocation.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: portfolio_allocation.png")
