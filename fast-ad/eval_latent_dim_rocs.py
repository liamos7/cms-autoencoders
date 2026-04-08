"""
Evaluate ROC/AUC for each AE model in outputs/latent_dim_variation/.

For each ae_zb_dim{N} directory:
  - Loads the AE with matching latent dim
  - Scores the held-out ZB test split (unseen 20%) as background
  - Scores each signal process as foreground
  - Computes per-signal ROC curves and AUC

Outputs:
  - plots/latent_dim_variation/roc_{signal}_{dim}.png  (per-signal, all dims overlaid)
  - plots/latent_dim_variation/auc_summary.png          (AUC vs latent dim, per signal)
  - plots/latent_dim_variation/auc_summary.csv          (raw numbers)
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
from sklearn.metrics import roc_curve, auc as sk_auc

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
DATA_DIR     = SCRIPT_DIR / "data" / "h5_files"
SWEEP_DIR    = SCRIPT_DIR / "outputs" / "latent_dim_variation"
PLOT_DIR     = SCRIPT_DIR / "plots" / "latent_dim_variation"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_EVENTS   = 200_000
BATCH_SIZE   = 2048

SIGNALS = {
    "SUEP":             "suep.h5",
    "GluGluH→ττ":      "glugluhtotautau.h5",
    "GluGluH→gg":      "glugluhtogg.h5",
    "H→2LL→4b":        "hto2longlivedto4b.h5",
    "VBF H→ττ":        "vbfhtotautau.h5",
    "VBF H→bb":        "vbfhto2b.h5",
    "Z'→ττ":           "zprimetotautau.h5",
    "ZZ":              "zz.h5",
    "tt":              "tt.h5",
    "Single Neutrino": "singleneutrino.h5",
}


# ── Helpers ────────────────────────────────────────────────────────────────────
def score_tensor(model, data_tensor):
    """Score a pre-built (N, 1, 18, 14) float32 tensor, return numpy scores."""
    scores = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(data_tensor), BATCH_SIZE):
            batch = data_tensor[i : i + BATCH_SIZE].to(DEVICE)
            scores.append(model.predict(batch).cpu().numpy())
    return np.concatenate(scores)


def load_h5_scores(model, filepath, max_events=MAX_EVENTS):
    """Load et_regions from h5, apply log-norm, return anomaly scores (numpy)."""
    with h5py.File(filepath, "r") as f:
        n = min(max_events, f["et_regions"].shape[0])
        data = f["et_regions"][:n].astype(np.float32)
    data = (np.log1p(data) / np.log1p(255)).astype(np.float32)
    data = torch.from_numpy(data).unsqueeze(1)  # (N, 1, 18, 14)

    scores = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(data), BATCH_SIZE):
            batch = data[i : i + BATCH_SIZE].to(DEVICE)
            s = model.predict(batch)
            scores.append(s.cpu().numpy())
    return np.concatenate(scores)


def load_h5_val_split(zb_path, data_dir, val_fraction=0.2, max_events=MAX_EVENTS):
    """
    Reproduce the exact 20% ZB val split from datasets.py.

    datasets.py concatenates ALL classes together and runs a single
    train_test_split(X, y, train_size=0.8, stratify=y, random_state=42).
    We replicate that here to get the exact same ZB val indices.
    """
    # Collect sizes of all class files in the same order as datasets.py
    class_files = [
        "zb", "glugluhtotautau", "glugluhtogg", "hto2longlivedto4b",
        "singleneutrino", "suep", "tt", "vbfhto2b", "vbfhtotautau",
        "zprimetotautau", "zz",
    ]
    sizes = {}
    for cls in class_files:
        p = data_dir / f"{cls}.h5"
        if p.exists():
            with h5py.File(p, "r") as f:
                sizes[cls] = f["et_regions"].shape[0]
        else:
            sizes[cls] = 0

    # Build label array (we only need y for the stratified split, not the data)
    label_map = {cls: i for i, cls in enumerate(class_files)}
    y_full = np.concatenate([
        np.full(sizes[cls], label_map[cls], dtype=np.int8) for cls in class_files if sizes[cls] > 0
    ])

    # StratifiedShuffleSplit only needs y, never materialises a copy of X
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, idx_val = next(sss.split(y_full, y_full))

    # ZB occupies the first sizes["zb"] positions in the concatenated array
    zb_size  = sizes["zb"]
    zb_val_idx = idx_val[idx_val < zb_size]          # val indices that belong to ZB
    zb_val_idx = np.sort(zb_val_idx)[:max_events]    # h5py requires sorted indices

    # Load the full ZB file in one contiguous read (fast), then index in numpy.
    # h5py fancy indexing with ~300k scattered indices is extremely slow because
    # it issues individual reads per index rather than one sequential memcpy.
    with h5py.File(zb_path, "r") as f:
        data = f["et_regions"][:].astype(np.float32)
    return data[zb_val_idx]


def score_array(model, data_np):
    """Score a (N, 18, 14) float32 numpy array after log-norm."""
    data = (np.log1p(data_np) / np.log1p(255)).astype(np.float32)
    data = torch.from_numpy(data).unsqueeze(1)

    scores = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(data), BATCH_SIZE):
            batch = data[i : i + BATCH_SIZE].to(DEVICE)
            s = model.predict(batch)
            scores.append(s.cpu().numpy())
    return np.concatenate(scores)


def discover_dims(sweep_dir):
    """Return sorted list of (latent_dim, model_path) from sweep_dir."""
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
                .replace("ℓ", "l").replace("→", "to"))


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    sys.path.insert(0, str(SCRIPT_DIR))
    from fastad.models import get_cicada_ae

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    dim_paths = discover_dims(SWEEP_DIR)
    if not dim_paths:
        print(f"No trained models found in {SWEEP_DIR}")
        sys.exit(1)

    dims = [d for d, _ in dim_paths]
    print(f"Found models for latent dims: {dims}")

    # ── Load ZB val split and all signals into RAM once ──────────────────────
    print("\nLoading data into RAM...")
    zb_path = DATA_DIR / "zb.h5"
    zb_val_data = load_h5_val_split(zb_path, DATA_DIR)
    zb_tensor = torch.from_numpy(
        (np.log1p(zb_val_data) / np.log1p(255)).astype(np.float32)
    ).unsqueeze(1)
    print(f"  ZB val events: {len(zb_val_data)}")
    del zb_val_data

    sig_tensors = {}
    for sig_name, fname in SIGNALS.items():
        path = DATA_DIR / fname
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping {sig_name}")
            continue
        with h5py.File(path, "r") as f:
            n = min(MAX_EVENTS, f["et_regions"].shape[0])
            data = f["et_regions"][:n].astype(np.float32)
        sig_tensors[sig_name] = torch.from_numpy(
            (np.log1p(data) / np.log1p(255)).astype(np.float32)
        ).unsqueeze(1)
        print(f"  {sig_name}: {n} events")

    # ── Score ZB + all signals per dim in a single model load ────────────────
    print("\nScoring...")
    bg_scores  = {}
    sig_scores = {name: {} for name in SIGNALS}

    for dim, ckpt_path in dim_paths:
        print(f"  dim={dim}...")
        model = get_cicada_ae(latent_dim=dim)
        ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        model.to(DEVICE)

        bg_scores[dim] = score_tensor(model, zb_tensor)
        for sig_name, tensor in sig_tensors.items():
            sig_scores[sig_name][dim] = score_tensor(model, tensor)

        del model
        torch.cuda.empty_cache()

    # ── Compute AUC table ─────────────────────────────────────────────────────
    print("\nComputing AUC...")
    records = []
    for sig_name in SIGNALS:
        for dim in dims:
            if dim not in sig_scores[sig_name]:
                continue
            bg = bg_scores[dim]
            sg = sig_scores[sig_name][dim]
            y_true  = np.concatenate([np.zeros(len(bg)), np.ones(len(sg))])
            y_score = np.concatenate([bg, sg])
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_val = sk_auc(fpr, tpr)
            records.append({"signal": sig_name, "latent_dim": dim, "auc": auc_val,
                            "fpr": fpr, "tpr": tpr})

    df_auc = pd.DataFrame([{k: v for k, v in r.items() if k not in ("fpr", "tpr")}
                            for r in records])
    csv_path = PLOT_DIR / "auc_summary.csv"
    df_auc.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    print(df_auc.pivot(index="signal", columns="latent_dim", values="auc").to_string())

    # ── Plot 1: per-signal ROC, all dims overlaid ─────────────────────────────
    print("\nPlotting per-signal ROC curves (all dims)...")
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.1)
    cmap = plt.colormaps.get_cmap("plasma")
    dim_colors = {d: cmap(i / max(len(dims) - 1, 1)) for i, d in enumerate(dims)}

    roc_lookup = {(r["signal"], r["latent_dim"]): (r["fpr"], r["tpr"]) for r in records}
    auc_lookup  = {(r["signal"], r["latent_dim"]):  r["auc"]            for r in records}

    for sig_name in SIGNALS:
        fig, ax = plt.subplots(figsize=(8, 6))
        for dim in dims:
            if (sig_name, dim) not in roc_lookup:
                continue
            fpr, tpr = roc_lookup[(sig_name, dim)]
            a = auc_lookup[(sig_name, dim)]
            ax.plot(fpr, tpr, color=dim_colors[dim], linewidth=1.8,
                    label=f"dim={dim}  (AUC={a:.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.2, linewidth=1)
        ax.set_xlabel("False Positive Rate (Zero Bias)")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"AE Latent Dim Sweep — {sig_name}")
        ax.legend(fontsize=8, loc="lower right")
        plt.tight_layout(pad=1.5)
        fname = f"roc_{safe_fname(sig_name)}_dim_sweep"
        for ext in ("png", "pdf"):
            fig.savefig(PLOT_DIR / f"{fname}.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}.png")

    # ── Plot 2: AUC vs latent dim, one line per signal ────────────────────────
    print("Plotting AUC summary...")
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = sns.color_palette("tab10", len(SIGNALS))
    for (sig_name, _), color in zip(SIGNALS.items(), palette):
        rows = df_auc[df_auc["signal"] == sig_name].sort_values("latent_dim")
        if rows.empty:
            continue
        ax.plot(rows["latent_dim"], rows["auc"], marker="o", linewidth=1.8,
                markersize=5, color=color, label=sig_name)
    ax.set_xlabel("Latent Dimension")
    ax.set_ylabel("AUC (ZB vs signal)")
    ax.set_title("AE AUC vs Latent Dimension — All Signals")
    ax.set_xticks(dims)
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout(pad=1.5)
    for ext in ("png", "pdf"):
        fig.savefig(PLOT_DIR / f"auc_summary.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: auc_summary.png")

    # ── Plot 3: mean AUC across signals vs latent dim ─────────────────────────
    mean_auc = df_auc.groupby("latent_dim")["auc"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(mean_auc["latent_dim"], mean_auc["auc"], marker="o", linewidth=2,
            markersize=6, color="steelblue")
    ax.set_xlabel("Latent Dimension")
    ax.set_ylabel("Mean AUC (across all signals)")
    ax.set_title("AE — Mean AUC vs Latent Dimension")
    ax.set_xticks(dims)
    plt.tight_layout(pad=1.5)
    for ext in ("png", "pdf"):
        fig.savefig(PLOT_DIR / f"mean_auc_vs_dim.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: mean_auc_vs_dim.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
