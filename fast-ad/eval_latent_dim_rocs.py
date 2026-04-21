"""
Evaluate ROC/AUC for each AE model in outputs/latent_dim_variation/.

Uses the held-out TEST split (10%) — the same three-way 80/10/10 stratified
split as ae_vs_nae_rocs.py — so checkpoint selection on val never touches
the data scored here.

For each ae_zb_dim{N} directory:
  - Loads the AE with matching latent dim
  - Scores the held-out ZB test split as background
  - Scores each signal process test split as foreground
  - Computes per-signal ROC curves and AUC (with 95% bootstrap CIs)

Outputs:
  - plots/latent_dim_variation/roc_{signal}_dim_sweep.png  (all dims overlaid)
  - plots/latent_dim_variation/auc_summary.png/.csv
  - plots/latent_dim_variation/mean_auc_vs_dim.png
"""

import os
import re
import sys
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from collections import OrderedDict
from sklearn.metrics import roc_curve, auc as sk_auc
from sklearn.model_selection import train_test_split

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
DATA_DIR   = SCRIPT_DIR / "data" / "h5_files"
SWEEP_DIR  = SCRIPT_DIR / "outputs" / "latent_dim_variation"
PLOT_DIR   = SCRIPT_DIR / "plots" / "latent_dim_variation"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2048
SPLIT      = "test"

# Must match datasets.py exactly
SEED_80_20 = 42
SEED_50_50 = 43

CLASS_ORDER = OrderedDict([
    ("zb",                0),
    ("glugluhtotautau",   1),
    ("glugluhtogg",       2),
    ("hto2longlivedto4b", 3),
    ("singleneutrino",    4),
    ("suep",              5),
    ("tt",                6),
    ("vbfhto2b",          7),
    ("vbfhtotautau",      8),
    ("zprimetotautau",    9),
    ("zz",               10),
])

PRETTY_NAMES = {
    "glugluhtotautau":   "GluGluH→ττ",
    "glugluhtogg":       "GluGluH→gg",
    "hto2longlivedto4b": "H→2LL→4b",
    "singleneutrino":    "Single Neutrino",
    "suep":              "SUEP",
    "tt":                "tt̄",
    "vbfhto2b":          "VBF H→bb",
    "vbfhtotautau":      "VBF H→ττ",
    "zprimetotautau":    "Z'→ττ",
    "zz":                "ZZ",
}

SIGNALS = [p for p in CLASS_ORDER if p != "zb"]


# ── Split helpers ──────────────────────────────────────────────────────────────
def get_split_indices_per_class(data_dir, split="test"):
    """
    Reproduce the three-way stratified split from datasets.CICADA on
    indices only (no data loaded). Same seeds, same stratified calls.

    Returns OrderedDict: process_name -> sorted array of local row indices
    for the requested split.
    """
    assert split in ("train", "val", "test")

    sizes = OrderedDict()
    for process in CLASS_ORDER:
        path = data_dir / f"{process}.h5"
        if path.exists():
            with h5py.File(path, "r") as f:
                sizes[process] = f["et_regions"].shape[0]
        else:
            sizes[process] = 0

    y_all = np.concatenate([
        np.full(s, lab, dtype=np.int32)
        for (p, lab), s in zip(CLASS_ORDER.items(), sizes.values()) if s > 0
    ])
    N = len(y_all)
    idx_all = np.arange(N, dtype=np.int32)

    idx_train, idx_rest, y_train, y_rest = train_test_split(
        idx_all, y_all,
        train_size=int(0.8 * N),
        stratify=y_all,
        random_state=SEED_80_20,
        shuffle=True,
    )

    if split == "train":
        idx_keep = idx_train
    else:
        idx_val, idx_test, _, _ = train_test_split(
            idx_rest, y_rest,
            test_size=0.5,
            stratify=y_rest,
            random_state=SEED_50_50,
            shuffle=True,
        )
        idx_keep = idx_val if split == "val" else idx_test

    del idx_all, y_all, idx_train, idx_rest, y_train, y_rest

    result = OrderedDict()
    offset = 0
    for process in CLASS_ORDER:
        s = sizes[process]
        if s == 0:
            continue
        mask = (idx_keep >= offset) & (idx_keep < offset + s)
        local = np.sort(idx_keep[mask] - offset)
        result[process] = local
        offset += s
    return result


# ── Scoring helpers ────────────────────────────────────────────────────────────
def load_and_score(model, filepath, row_indices):
    """
    Read only the requested rows via contiguous chunk reads + boolean masking,
    apply log-norm, and return anomaly scores.
    """
    with h5py.File(filepath, "r") as f:
        total = f["et_regions"].shape[0]

    keep = np.zeros(total, dtype=bool)
    keep[row_indices] = True

    norm_factor = np.float32(np.log1p(255))
    scores = []

    with h5py.File(filepath, "r") as f:
        ds = f["et_regions"]
        CHUNK = 50_000
        for start in range(0, total, CHUNK):
            end = min(start + CHUNK, total)
            chunk_keep = keep[start:end]
            if not chunk_keep.any():
                continue
            raw = ds[start:end][chunk_keep]
            for i in range(0, len(raw), BATCH_SIZE):
                arr = raw[i : i + BATCH_SIZE].astype(np.float32)
                np.log1p(arr, out=arr)
                arr /= norm_factor
                t = torch.from_numpy(arr).unsqueeze(1).to(DEVICE)
                with torch.no_grad():
                    s = model.predict(t)
                scores.append(s.cpu().numpy())
                del t
            del raw

    return np.concatenate(scores) if scores else np.array([], dtype=np.float32)


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
        aucs[i] = sk_auc(fpr, tpr)
    aucs = aucs[~np.isnan(aucs)]
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi)


# ── Misc helpers ───────────────────────────────────────────────────────────────
def discover_dims(sweep_dir):
    pattern = re.compile(r"ae_zb_dim(\d+)$")
    results = []
    for d in sorted(sweep_dir.iterdir()):
        m = pattern.match(d.name)
        if m and (d / "model_best.pkl").exists():
            results.append((int(m.group(1)), d / "model_best.pkl"))
    return sorted(results)


def safe_fname(name):
    return (name.replace(" ", "_").replace("→", "to")
                .replace("'", "").replace("τ", "tau")
                .replace("ℓ", "l").replace("→", "to").replace("̄", ""))


def save_fig(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(PLOT_DIR / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    print(f"  Saved: {stem}.png")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    sys.path.insert(0, str(SCRIPT_DIR))
    from fastad.models import get_cicada_ae

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.1)

    dim_paths = discover_dims(SWEEP_DIR)
    if not dim_paths:
        print(f"No trained models found in {SWEEP_DIR}")
        sys.exit(1)

    dims = [d for d, _ in dim_paths]
    print(f"Found models for latent dims: {dims}")
    print(f"Evaluating on split: {SPLIT!r}")

    # ── Compute test-split indices once (no data loaded) ─────────────────────
    print(f"\nComputing {SPLIT}-split indices...")
    split_indices = get_split_indices_per_class(DATA_DIR, split=SPLIT)
    for p, idx in split_indices.items():
        print(f"  {PRETTY_NAMES.get(p, p):<20} {len(idx):>8} events")

    # ── Score all processes for each dim ─────────────────────────────────────
    print("\nScoring...")
    bg_scores  = {}   # dim -> np array
    sig_scores = {p: {} for p in SIGNALS}

    for dim, ckpt_path in dim_paths:
        print(f"  dim={dim}...")
        model = get_cicada_ae(latent_dim=dim)
        ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        model.to(DEVICE).eval()

        for process, local_idx in split_indices.items():
            filepath = DATA_DIR / f"{process}.h5"
            if not filepath.exists():
                continue
            scores = load_and_score(model, filepath, local_idx)
            if process == "zb":
                bg_scores[dim] = scores
            else:
                sig_scores[process][dim] = scores

        del model
        torch.cuda.empty_cache()

    # ── Compute AUC table ─────────────────────────────────────────────────────
    print("\nComputing AUC...")
    records = []
    for process in SIGNALS:
        pretty = PRETTY_NAMES.get(process, process)
        for dim in dims:
            if dim not in sig_scores[process] or dim not in bg_scores:
                continue
            bg = bg_scores[dim]
            sg = sig_scores[process][dim]
            y_true  = np.concatenate([np.zeros(len(bg)), np.ones(len(sg))])
            y_score = np.concatenate([bg, sg])
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_val = sk_auc(fpr, tpr)
            ci = bootstrap_auc_ci(y_true, y_score, n_boot=200, seed=dim)
            records.append({
                "signal": pretty, "latent_dim": dim,
                "auc": auc_val, "ci_lo": ci[0], "ci_hi": ci[1],
                "fpr": fpr, "tpr": tpr,
            })

    df_auc = pd.DataFrame([{k: v for k, v in r.items() if k not in ("fpr", "tpr")}
                            for r in records])
    csv_path = PLOT_DIR / "auc_summary.csv"
    df_auc.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    print(df_auc.pivot(index="signal", columns="latent_dim", values="auc").to_string())

    # ── Plot 1: per-signal ROC, all dims overlaid ─────────────────────────────
    print("\nPlotting per-signal ROC curves (all dims)...")
    cmap = plt.colormaps.get_cmap("plasma")
    dim_colors = {d: cmap(i / max(len(dims) - 1, 1)) for i, d in enumerate(dims)}

    roc_lookup = {(r["signal"], r["latent_dim"]): (r["fpr"], r["tpr"]) for r in records}
    auc_lookup  = {(r["signal"], r["latent_dim"]): (r["auc"], r["ci_lo"], r["ci_hi"])
                   for r in records}

    for process in SIGNALS:
        pretty = PRETTY_NAMES.get(process, process)
        fig, ax = plt.subplots(figsize=(8, 6))
        for dim in dims:
            if (pretty, dim) not in roc_lookup:
                continue
            fpr, tpr = roc_lookup[(pretty, dim)]
            a, lo, hi = auc_lookup[(pretty, dim)]
            ax.plot(fpr, tpr, color=dim_colors[dim], linewidth=1.8,
                    label=f"dim={dim}  AUC={a:.3f} [{lo:.3f}, {hi:.3f}]")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.2, linewidth=1)
        ax.set_xscale("log")
        ax.set_xlabel("False Positive Rate (Zero Bias)")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"AE Latent Dim Sweep — {pretty} ({SPLIT} split)")
        ax.legend(fontsize=8, loc="lower right")
        plt.tight_layout(pad=1.5)
        save_fig(fig, f"roc_{safe_fname(pretty)}_dim_sweep")

    # ── Plot 2: AUC vs latent dim, one line per signal ────────────────────────
    print("Plotting AUC summary...")
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = sns.color_palette("tab10", len(SIGNALS))
    for process, color in zip(SIGNALS, palette):
        pretty = PRETTY_NAMES.get(process, process)
        rows = df_auc[df_auc["signal"] == pretty].sort_values("latent_dim")
        if rows.empty:
            continue
        ax.plot(rows["latent_dim"], rows["auc"], marker="o", linewidth=1.8,
                markersize=5, color=color, label=pretty)
    ax.set_xlabel("Latent Dimension")
    ax.set_ylabel("AUC (ZB vs signal)")
    ax.set_title(f"AE AUC vs Latent Dimension — All Signals ({SPLIT} split)")
    ax.set_xticks(dims)
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout(pad=1.5)
    save_fig(fig, "auc_summary")

    # ── Plot 3: mean AUC across signals vs latent dim ─────────────────────────
    mean_auc = df_auc.groupby("latent_dim")["auc"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(mean_auc["latent_dim"], mean_auc["auc"], marker="o", linewidth=2,
            markersize=6, color="steelblue")
    ax.set_xlabel("Latent Dimension")
    ax.set_ylabel("Mean AUC (across all signals)")
    ax.set_title(f"AE — Mean AUC vs Latent Dimension ({SPLIT} split)")
    ax.set_xticks(dims)
    plt.tight_layout(pad=1.5)
    save_fig(fig, "mean_auc_vs_dim")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print(f"Results on {SPLIT} split (held out during training)")
    print("=" * 90)
    print(f"{'Signal':<20} {'Dim':>5} {'AUC':>8} {'95% CI':>24}")
    print("-" * 90)
    for _, row in df_auc.sort_values(["signal", "latent_dim"]).iterrows():
        ci_s = f"[{row['ci_lo']:.4f}, {row['ci_hi']:.4f}]"
        print(f"{row['signal']:<20} {int(row['latent_dim']):>5} {row['auc']:>8.4f} {ci_s:>24}")
    print("=" * 90)

    print("\nDone.")


if __name__ == "__main__":
    main()
