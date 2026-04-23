"""
nae_mc_oracle_rocs.py — ROC curves for nae_mc_oracle
=====================================================

Training regime:  model sees ZB-only as inliers; ZB+SingleNeutrino as MC negatives
Background (ROC): Zero Bias only
Signals (ROC):    All 10 signal processes including Single Neutrino
                  (SN is unseen during training — tests out-of-distribution sensitivity)
Model:            outputs/nae_mc_oracle_dim-20_fixed-zb/model_best.pkl
Split:            test (10%, held out during all training)

Memory-efficient: contiguous chunk reads + boolean masking, one class at a time.
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
from pathlib import Path
from collections import OrderedDict
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

from fastad.models.__init__ import get_cicada_nae_with_energy

sns.set_theme(style='whitegrid', context='notebook', font_scale=1.1)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR  = Path("./data/h5_files/")
NAE_PATH  = "outputs/nae_mc_oracle_dim-20_fixed-zb/model_best.pkl"
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH     = 1024
OUT_DIR   = "plots/rocs_mc_oracle"
os.makedirs(OUT_DIR, exist_ok=True)

# Must match datasets.py exactly
SEED_80_20 = 42
SEED_50_50 = 43
SPLIT = "test"

# ── Class ordering (must match datasets.py exactly) ───────────────────────────
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

# ZB-only background. SN appears as a signal (unseen by the model during training —
# the MC negatives exposed it during EBM training but it was never an inlier).
BACKGROUND_PROCESSES = {"zb"}

PRETTY_NAMES = {
    "zb":                "Zero Bias",
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


def get_split_indices_per_class(data_dir, split="test"):
    """
    Reproduce the three-way stratified split from datasets.CICADA on indices
    only (no data loaded). Same seeds, same stratified calls, same result.

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

    train_size = int(0.8 * N)
    idx_train, idx_rest, y_train, y_rest = train_test_split(
        idx_all, y_all,
        train_size=train_size,
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


def load_and_score(model, filepath, row_indices):
    """
    Read only the requested rows from an HDF5 file and score them.
    Uses contiguous chunk reads + boolean masking (fast for large arrays).
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
            for i in range(0, len(raw), BATCH):
                arr = raw[i:i+BATCH].astype(np.float32)
                np.log1p(arr, out=arr)
                arr /= norm_factor
                t = torch.from_numpy(arr).unsqueeze(1).to(DEVICE)
                with torch.no_grad():
                    s = model.predict(t)
                scores.append(s.cpu().numpy())
                del t
            del raw

    return np.concatenate(scores) if scores else np.array([], dtype=np.float32)


def load_teacher_scores(filepath, row_indices):
    """Read pre-computed teacher_score values directly from HDF5."""
    with h5py.File(filepath, "r") as f:
        total = f["teacher_score"].shape[0]
        keep = np.zeros(total, dtype=bool)
        keep[row_indices] = True
        scores = np.array(f["teacher_score"])[keep]
    return scores.astype(np.float32)


def bootstrap_auc_ci(y_true, scores, n_boot=200, seed=0, max_samples=50_000):
    """Non-parametric bootstrap 95% CI for AUC."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    # Subsample once to keep bootstrap fast on large arrays
    if n > max_samples:
        sub = rng.choice(n, max_samples, replace=False)
        y_true = y_true[sub]
        scores = scores[sub]
        n = max_samples
    aucs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        if y_true[idx].min() == y_true[idx].max():
            aucs[i] = np.nan
            continue
        fpr_b, tpr_b, _ = roc_curve(y_true[idx], scores[idx])
        aucs[i] = auc(fpr_b, tpr_b)
    aucs = aucs[~np.isnan(aucs)]
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi)


def safe_fname(name):
    return (name.replace(' ', '_').replace('→', 'to')
                .replace("'", '').replace('τ', 'tau').replace('ℓ', 'l')
                .replace('̄', ''))


def save_fig(fig, stem):
    fig.savefig(f'{OUT_DIR}/{stem}.png', dpi=200, bbox_inches='tight')
    fig.savefig(f'{OUT_DIR}/{stem}.pdf', bbox_inches='tight')
    print(f"  Saved: {OUT_DIR}/{stem}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

print(f"Device: {DEVICE}", flush=True)
print(f"Model:  {NAE_PATH}", flush=True)
print(f"Split:  {SPLIT!r}", flush=True)
print(f"Background: Zero Bias only (Single Neutrino treated as signal)", flush=True)

print("\nLoading model...", flush=True)
model_nae = get_cicada_nae_with_energy(NAE_PATH).to(DEVICE).eval()

print(f"\nComputing {SPLIT}-split indices (deterministic, no data loaded)...", flush=True)
split_indices = get_split_indices_per_class(DATA_DIR, split=SPLIT)
for p, idx in split_indices.items():
    tag = "[BG]" if p in BACKGROUND_PROCESSES else "[SIG]"
    print(f"  {tag} {PRETTY_NAMES.get(p, p):<20} {len(idx):>8} events", flush=True)

print("\nScoring...", flush=True)
bg_scores_nae     = []
bg_scores_teacher = []
signal_results    = OrderedDict()  # name -> (scores_nae, scores_teacher)

for process, local_idx in split_indices.items():
    pretty   = PRETTY_NAMES.get(process, process)
    filepath = DATA_DIR / f"{process}.h5"
    print(f"  {pretty} ({len(local_idx)} events)...", end=" ", flush=True)

    s_nae     = load_and_score(model_nae, filepath, local_idx)
    s_teacher = load_teacher_scores(filepath, local_idx)
    print("done")

    if process in BACKGROUND_PROCESSES:
        bg_scores_nae.append(s_nae)
        bg_scores_teacher.append(s_teacher)
    else:
        signal_results[pretty] = (s_nae, s_teacher)

bg_nae     = np.concatenate(bg_scores_nae)
bg_teacher = np.concatenate(bg_scores_teacher)

print(f"\nBackground (ZB only, {SPLIT}): {len(bg_nae)} events", flush=True)
for name, (s, _) in signal_results.items():
    print(f"  {name}: {len(s)} events", flush=True)


# ── Per-signal ROC plots ───────────────────────────────────────────────────────
print("\nPlotting per-signal ROC curves (with 95% bootstrap CIs)...", flush=True)
palette   = sns.color_palette("tab10", len(signal_results))
auc_cache = {}

for (name, (s_nae, s_teacher)), color in zip(signal_results.items(), palette):
    y_true          = np.concatenate([np.zeros(len(bg_nae)), np.ones(len(s_nae))])
    scores_nae      = np.concatenate([bg_nae,     s_nae])
    scores_teacher  = np.concatenate([bg_teacher, s_teacher])

    fpr_nae,     tpr_nae,     _ = roc_curve(y_true, scores_nae)
    fpr_teacher, tpr_teacher, _ = roc_curve(y_true, scores_teacher)
    auc_nae     = auc(fpr_nae,     tpr_nae)
    auc_teacher = auc(fpr_teacher, tpr_teacher)

    ci_nae     = bootstrap_auc_ci(y_true, scores_nae,     n_boot=200, seed=0)
    ci_teacher = bootstrap_auc_ci(y_true, scores_teacher, n_boot=200, seed=1)
    auc_cache[name] = (auc_nae, ci_nae, auc_teacher, ci_teacher)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr_teacher, tpr_teacher, linestyle=':', linewidth=1.8, color='gray',
            label=f'Teacher  AUC = {auc_teacher:.4f} [{ci_teacher[0]:.4f}, {ci_teacher[1]:.4f}]')
    ax.plot(fpr_nae, tpr_nae, linestyle='-', linewidth=2.2, color=color,
            label=f'NAE (oracle)  AUC = {auc_nae:.4f} [{ci_nae[0]:.4f}, {ci_nae[1]:.4f}]')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.2, linewidth=1)
    ax.set_xscale('log')
    ax.set_xlabel('False Positive Rate (Zero Bias)')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC — {name} ({SPLIT} split)\nBG: Zero Bias only')
    ax.legend(fontsize=9)
    plt.tight_layout(pad=1.5)
    save_fig(fig, f'roc_{safe_fname(name)}_{SPLIT}')


# ── Combined NAE-oracle plot ──────────────────────────────────────────────────
print("\nPlotting combined NAE oracle ROC...", flush=True)
fig, ax = plt.subplots(figsize=(10, 8))
for (name, (s_nae, _)), color in zip(signal_results.items(), palette):
    y_true = np.concatenate([np.zeros(len(bg_nae)), np.ones(len(s_nae))])
    fpr, tpr, _ = roc_curve(y_true, np.concatenate([bg_nae, s_nae]))
    ax.plot(fpr, tpr, linewidth=2, color=color,
            label=f'{name} (AUC = {auc(fpr, tpr):.3f})')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.2, linewidth=1)
ax.set_xscale('log')
ax.set_xlabel('False Positive Rate (Zero Bias)')
ax.set_ylabel('True Positive Rate')
ax.set_title(f'NAE MC Oracle — All Signals\n({SPLIT} split, BG: Zero Bias only)')
ax.legend(fontsize=9, loc='lower right')
plt.tight_layout(pad=1.5)
save_fig(fig, f'roc_nae_oracle_all_{SPLIT}')


# ── Combined NAE-oracle vs Teacher overlay ────────────────────────────────────
print("\nPlotting combined NAE oracle vs Teacher ROC...", flush=True)
fig, ax = plt.subplots(figsize=(10, 8))
for (name, (s_nae, s_teacher)), color in zip(signal_results.items(), palette):
    y_true = np.concatenate([np.zeros(len(bg_nae)), np.ones(len(s_nae))])
    fpr_nae,     tpr_nae,     _ = roc_curve(y_true, np.concatenate([bg_nae,     s_nae]))
    fpr_teacher, tpr_teacher, _ = roc_curve(y_true, np.concatenate([bg_teacher, s_teacher]))
    ax.plot(fpr_teacher, tpr_teacher, linestyle=':', linewidth=1.2, color=color, alpha=0.5)
    ax.plot(fpr_nae,     tpr_nae,     linestyle='-', linewidth=2.0, color=color, label=name)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.2, linewidth=1)
from matplotlib.lines import Line2D
style_legend = [
    Line2D([0], [0], color='gray', linestyle=':', linewidth=1.5, label='Teacher (dotted)'),
    Line2D([0], [0], color='gray', linestyle='-', linewidth=2.0, label='NAE oracle (solid)'),
]
signal_handles = [
    Line2D([0], [0], color=color, linewidth=2, label=name)
    for name, color in zip(signal_results.keys(), palette)
]
ax.legend(handles=style_legend + signal_handles, fontsize=8, loc='lower right', ncol=2)
ax.set_xscale('log')
ax.set_xlabel('False Positive Rate (Zero Bias)')
ax.set_ylabel('True Positive Rate')
ax.set_title(f'Teacher (dotted) vs NAE Oracle (solid)\n({SPLIT} split, BG: Zero Bias only)')
plt.tight_layout(pad=1.5)
save_fig(fig, f'roc_oracle_vs_teacher_all_{SPLIT}')


# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print(f"Results on {SPLIT} split — background: Zero Bias only (Single Neutrino is a signal)")
print("=" * 90)
print(f"{'Signal':<20} {'NAE AUC':>10} {'NAE 95% CI':>22} "
      f"{'Teacher AUC':>12} {'Teacher 95% CI':>22} {'Δ(NAE-T)':>10}")
print("-" * 90)
for name, (a_nae, ci_nae, a_teacher, ci_teacher) in auc_cache.items():
    delta     = a_nae - a_teacher
    ci_nae_s  = f"[{ci_nae[0]:.4f}, {ci_nae[1]:.4f}]"
    ci_t_s    = f"[{ci_teacher[0]:.4f}, {ci_teacher[1]:.4f}]"
    print(f"{name:<20} {a_nae:>10.4f} {ci_nae_s:>22} "
          f"{a_teacher:>12.4f} {ci_t_s:>22} {delta:>+10.4f}")
print("=" * 90)

print("\nDone.")
