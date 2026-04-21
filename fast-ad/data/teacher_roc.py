"""
ROC curves for teacher_score: each signal dataset vs Zero Bias background.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc

LABEL_MAP = {
    "glugluhtogg":         r"$gg \to H \to \gamma\gamma$",
    "glugluhtotautau":     r"$gg \to H \to \tau\tau$",
    "hto2longlivedto4b":   r"$H \to 2LL \to 4b$",
    "singleneutrino":      "Single Neutrino",
    "suep":                "SUEP",
    "tt":                  r"$t\bar{t}$",
    "vbfhto2b":            r"VBF $H \to bb$",
    "vbfhtotautau":        r"VBF $H \to \tau\tau$",
    "zprimetotautau":      r"$Z' \to \tau\tau$",
    "zz":                  r"$ZZ$",
}

COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#469990", "#dcbeff",
]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})


MAX_EVENTS = 200_000


def load_scores(filepath):
    with h5py.File(filepath, "r") as f:
        n = min(MAX_EVENTS, f["teacher_score"].shape[0])
        scores = f["teacher_score"][:n].flatten().astype(np.float32)
    return scores[np.isfinite(scores)]


def bootstrap_auc_ci(y_true, scores, n_boot=200, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        if y_true[idx].min() == y_true[idx].max():
            aucs[i] = np.nan
            continue
        fpr, tpr, _ = roc_curve(y_true[idx], scores[idx])
        aucs[i] = auc(fpr, tpr)
    aucs = aucs[~np.isnan(aucs)]
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    h5_dir = Path("h5_files")
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)

    zb_path = h5_dir / "zb.h5"
    if not zb_path.exists():
        raise FileNotFoundError(f"Zero Bias file not found: {zb_path}")

    zb_scores = load_scores(zb_path)
    print(f"Zero Bias: {len(zb_scores):,} events")

    signal_files = sorted(
        p for p in h5_dir.glob("*.h5") if p.stem != "zb"
    )

    # ── Score all signals and compute ROC + CI once ───────────────────────────
    results = []
    for i, filepath in enumerate(signal_files):
        stem = filepath.stem
        label = LABEL_MAP.get(stem, stem)
        color = COLORS[i % len(COLORS)]

        sig_scores = load_scores(filepath)
        print(f"{label:35s}  n={len(sig_scores):,}")

        y_score = np.concatenate([sig_scores, zb_scores])
        y_true  = np.concatenate([np.ones(len(sig_scores)), np.zeros(len(zb_scores))])

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        ci_lo, ci_hi = bootstrap_auc_ci(y_true, y_score, n_boot=200, seed=i)

        results.append((label, color, fpr, tpr, roc_auc, ci_lo, ci_hi))
        print(f"  AUC={roc_auc:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")

    # ── Combined ROC plot ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7))
    for label, color, fpr, tpr, roc_auc, ci_lo, ci_hi in results:
        ax.plot(fpr, tpr, color=color, linewidth=1.8,
                label=f"{label}  AUC={roc_auc:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random")
    ax.set_xlabel("False Positive Rate (ZB pass rate)")
    ax.set_ylabel("True Positive Rate (Signal efficiency)")
    ax.set_title("Teacher Score ROC: signal vs Zero Bias")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", framealpha=0.9, edgecolor="0.7")
    fig.tight_layout()
    out = output_dir / "teacher_roc.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}")

    # ── Log-x ROC (zoom into low FPR region) ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7))
    for label, color, fpr, tpr, roc_auc, ci_lo, ci_hi in results:
        mask = fpr > 0
        ax.plot(fpr[mask], tpr[mask], color=color, linewidth=1.8,
                label=f"{label}  AUC={roc_auc:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")
    ax.set_xscale("log")
    ax.set_xlabel("False Positive Rate (ZB pass rate)")
    ax.set_ylabel("True Positive Rate (Signal efficiency)")
    ax.set_title("Teacher Score ROC (log FPR): signal vs Zero Bias")
    ax.set_xlim(1e-4, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25, linestyle="--", which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", framealpha=0.9, edgecolor="0.7")
    fig.tight_layout()
    out = output_dir / "teacher_roc_logx.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
