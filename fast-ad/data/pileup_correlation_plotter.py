"""
Plot Pearson correlation and R^2 between pileup (nPV) and total_et
for each event type, as a bar chart.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

LABEL_MAP = {
    "glugluhtogg":         r"$gg \to H \to \gamma\gamma$",
    "glugluhtotautau":     r"$gg \to H \to \tau\tau$",
    "hto2longlivedto4b":   r"$H \to 2LL \to 4b$",
    "singleneutrino":      "Single Neutrino",
    "suep":                "SUEP",
    "tt":                  r"$t\bar{t}$",
    "vbfhto2b":            r"VBF $H \to bb$",
    "vbfhtotautau":        r"VBF $H \to \tau\tau$",
    "zb":                  "Zero Bias",
    "zprimetotautau":      r"$Z' \to \tau\tau$",
    "zz":                  r"$ZZ$",
}

COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#469990", "#dcbeff", "#800000",
]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 11,
    "figure.dpi": 150,
})


def load_pair(filepath):
    with h5py.File(filepath, "r") as f:
        if "nPV" not in f or "total_et" not in f:
            return None, None
        nPV = np.array(f["nPV"]).flatten().astype(float)
        total_et = np.array(f["total_et"]).flatten().astype(float)
    # Drop events where either is NaN or non-finite
    mask = np.isfinite(nPV) & np.isfinite(total_et)
    return nPV[mask], total_et[mask]


def scatter_panel(ax, nPV, total_et, label, color, r, r2,
                  xlim=None, ylim=None):
    # Subsample for speed/clarity
    n = len(nPV)
    max_pts = 5000
    if n > max_pts:
        idx = np.random.default_rng(0).choice(n, max_pts, replace=False)
        nPV_plot, et_plot = nPV[idx], total_et[idx]
    else:
        nPV_plot, et_plot = nPV, total_et

    ax.scatter(nPV_plot, et_plot, s=2, alpha=0.25, color=color, rasterized=True)

    # Regression line over shared x range
    x_lo, x_hi = xlim if xlim else (nPV.min(), nPV.max())
    x_range = np.linspace(x_lo, x_hi, 200)
    slope, intercept, *_ = stats.linregress(nPV, total_et)
    ax.plot(x_range, slope * x_range + intercept, color="k", linewidth=1.2)

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    ax.set_title(label, fontsize=10, pad=3)
    ax.set_xlabel("nPV (pileup)", fontsize=9)
    ax.set_ylabel("Total ET [GeV]", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.text(
        0.97, 0.97,
        f"r = {r:.3f}\n$R^2$ = {r2:.3f}",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    h5_dir = Path("h5_files")
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)

    files = sorted(h5_dir.glob("*.h5"))
    results = []

    for filepath in files:
        stem = filepath.stem
        label = LABEL_MAP.get(stem, stem)
        nPV, total_et = load_pair(filepath)
        if nPV is None or len(nPV) < 2:
            print(f"Skipping {stem}: missing data")
            continue
        r, pval = stats.pearsonr(nPV, total_et)
        r2 = r ** 2
        results.append({"stem": stem, "label": label, "r": r, "r2": r2,
                         "nPV": nPV, "total_et": total_et})
        print(f"{label:35s}  r={r:.4f}  R²={r2:.4f}  n={len(nPV):,}")

    if not results:
        print("No data loaded.")
        return

    # ── Bar chart: r and R² side by side ──────────────────────────────────────
    n = len(results)
    labels = [d["label"] for d in results]
    rs = [d["r"] for d in results]
    r2s = [d["r2"] for d in results]
    x = np.arange(n)
    w = 0.38

    fig, ax = plt.subplots(figsize=(max(10, n * 1.1), 5))
    bars_r = ax.bar(x - w / 2, rs,  width=w, label="Pearson r",  color="#4363d8", alpha=0.85)
    bars_r2 = ax.bar(x + w / 2, r2s, width=w, label=r"$R^2$", color="#e6194b", alpha=0.85)

    for bar in (*bars_r, *bars_r2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Correlation")
    ax.set_title("Pileup (nPV) vs Total ET: Pearson $r$ and $R^2$ per event type")
    ax.set_ylim(0, min(1.15, max(*rs, *r2s) * 1.25))
    ax.axhline(0, color="k", linewidth=0.6)
    ax.legend()
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = output_dir / "pileup_correlation_bar.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}")

    # ── Scatter panels: one per event type ────────────────────────────────────
    # Compute shared axis limits across all event types
    all_nPV = np.concatenate([d["nPV"] for d in results])
    all_et  = np.concatenate([d["total_et"] for d in results])
    xlim = (np.percentile(all_nPV, 0.5), np.percentile(all_nPV, 99.5))
    ylim = (np.percentile(all_et,  0.5), np.percentile(all_et,  99.5))

    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
    axes = axes.flatten()

    for i, (d, color) in enumerate(zip(results, COLORS)):
        scatter_panel(axes[i], d["nPV"], d["total_et"],
                      d["label"], color, d["r"], d["r2"],
                      xlim=xlim, ylim=ylim)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Pileup (nPV) vs Total ET per event type", fontsize=14, y=1.01)
    fig.tight_layout()
    out = output_dir / "pileup_correlation_scatter.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
