import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
from pathlib import Path
from sklearn.metrics import roc_curve, auc
from fastad.models.__init__ import get_cicada_ae, get_cicada_nae_with_energy

sns.set_theme(style='whitegrid', context='notebook', font_scale=1.1)

DATA_DIR = Path("./data/h5_files/")
AE_PATH  = "outputs/ae_zb_npv_geq10_dim20/model_best.pkl"
NAE_PATH = "outputs/nae_phase2_tuned_dim20/model_best.pkl"
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Max events to load per file (keeps memory manageable)
MAX_EVENTS = 200_000

# ── Load models ──────────────────────────────────────────────────────────────
print("Loading models...")
model_ae = get_cicada_ae()
ae_checkpoint = torch.load(AE_PATH, map_location=DEVICE, weights_only=False)
model_ae.load_state_dict(ae_checkpoint['model_state'])
model_ae.to(DEVICE).eval()

model_nae = get_cicada_nae_with_energy(NAE_PATH).to(DEVICE).eval()


# ── ZB val-split helper ───────────────────────────────────────────────────────
def get_zb_val_indices(data_dir, max_events=MAX_EVENTS):
    """
    Reproduce the exact ZB indices in the 20% val split from datasets.py.

    datasets.py concatenates all classes in order and runs a single
    train_test_split(random_state=42, stratify=y) over the whole array.
    We replicate that to identify which ZB rows end up in the val set.
    """
    class_files = [
        "zb", "glugluhtotautau", "glugluhtogg", "hto2longlivedto4b",
        "singleneutrino", "suep", "tt", "vbfhto2b", "vbfhtotautau",
        "zprimetotautau", "zz",
    ]
    sizes = {}
    for cls in class_files:
        p = data_dir / f"{cls}.h5"
        sizes[cls] = 0
        if p.exists():
            with h5py.File(p, "r") as f:
                sizes[cls] = f["et_regions"].shape[0]

    y_full = np.concatenate([
        np.full(sizes[cls], i, dtype=np.int8) for i, cls in enumerate(class_files) if sizes[cls] > 0
    ])

    # StratifiedShuffleSplit only needs y, never materialises a copy of X
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, idx_val = next(sss.split(y_full, y_full))

    zb_size     = sizes["zb"]
    zb_val_idx  = idx_val[idx_val < zb_size]
    return np.sort(zb_val_idx)[:max_events]    # h5py requires sorted indices


# ── Shared scoring helper ─────────────────────────────────────────────────────
def score_h5(model, filepath, max_events=MAX_EVENTS):
    """Load et_regions from an h5 file, apply log-norm, and return anomaly scores."""
    with h5py.File(filepath, 'r') as f:
        n = min(max_events, f['et_regions'].shape[0])
        data = f['et_regions'][:n]
    data = (np.log1p(data.astype(np.float32)) / np.log1p(255)).astype(np.float32)
    data = torch.from_numpy(data).unsqueeze(1)  # (N, 1, 18, 14)

    scores = []
    with torch.no_grad():
        for i in range(0, len(data), 1024):
            batch = data[i:i+1024].to(DEVICE)
            s = model.predict(batch) if hasattr(model, 'predict') else model.reconstruction_loss(batch)
            scores.append(s.cpu().numpy())
    return np.concatenate(scores)


# ── Background scores (ZB val split only — no leakage) ───────────────────────
print("Scoring background (ZB val split)...")
zb_path     = DATA_DIR / "zb.h5"
zb_val_idx  = get_zb_val_indices(DATA_DIR)
with h5py.File(zb_path, "r") as f:
    zb_val_data = f["et_regions"][:].astype(np.float32)
zb_val_data = zb_val_data[zb_val_idx]
zb_val_data = (np.log1p(zb_val_data) / np.log1p(255)).astype(np.float32)
zb_tensor   = torch.from_numpy(zb_val_data).unsqueeze(1)

def score_tensor(model, data_tensor):
    scores = []
    with torch.no_grad():
        for i in range(0, len(data_tensor), 1024):
            batch = data_tensor[i:i+1024].to(DEVICE)
            s = model.predict(batch) if hasattr(model, 'predict') else model.reconstruction_loss(batch)
            scores.append(s.cpu().numpy())
    return np.concatenate(scores)

bg_ae_scores  = score_tensor(model_ae,  zb_tensor)
bg_nae_scores = score_tensor(model_nae, zb_tensor)
print(f"  ZB val events scored: {len(bg_ae_scores)}")


# ── Signal files ──────────────────────────────────────────────────────────────
signals = {
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


# ── Score all signals ─────────────────────────────────────────────────────────
print("Scoring signals...")
signal_results = {}
for name, fname in signals.items():
    path = DATA_DIR / fname
    if not path.exists():
        print(f"  WARNING: {path} not found, skipping {name}")
        continue
    print(f"  {name}...")
    signal_results[name] = (score_h5(model_ae, path), score_h5(model_nae, path))


# ── Plotting helpers ──────────────────────────────────────────────────────────
os.makedirs("plots", exist_ok=True)

def safe_fname(name):
    return (name.replace(' ', '_').replace('→', 'to')
                .replace("'", '').replace('τ', 'tau').replace('ℓ', 'l'))

def save_fig(fig, stem):
    fig.savefig(f'plots/{stem}.png', dpi=200, bbox_inches='tight')
    fig.savefig(f'plots/{stem}.pdf', bbox_inches='tight')
    print(f"  Saved: plots/{stem}.png")
    plt.close(fig)


# ── Per-signal plots (AE vs NAE) ──────────────────────────────────────────────
print("Plotting per-signal ROC curves...")
palette = sns.color_palette("tab10", len(signal_results))

for (name, (s_ae, s_nae)), color in zip(signal_results.items(), palette):
    fig, ax = plt.subplots(figsize=(7, 6))

    y_true_ae  = np.concatenate([np.zeros(len(bg_ae_scores)),  np.ones(len(s_ae))])
    y_true_nae = np.concatenate([np.zeros(len(bg_nae_scores)), np.ones(len(s_nae))])

    fpr_ae,  tpr_ae,  _ = roc_curve(y_true_ae,  np.concatenate([bg_ae_scores,  s_ae]))
    fpr_nae, tpr_nae, _ = roc_curve(y_true_nae, np.concatenate([bg_nae_scores, s_nae]))

    ax.plot(fpr_ae,  tpr_ae,  linestyle='--', linewidth=1.8, color=color, alpha=0.7,
            label=f'AE  (AUC = {auc(fpr_ae,  tpr_ae):.3f})')
    ax.plot(fpr_nae, tpr_nae, linestyle='-',  linewidth=2.2, color=color,
            label=f'NAE (AUC = {auc(fpr_nae, tpr_nae):.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.2, linewidth=1)

    ax.set_xlabel('False Positive Rate (Zero Bias)')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC — {name}')
    ax.legend()
    plt.tight_layout(pad=1.5)
    save_fig(fig, f'roc_{safe_fname(name)}')


# ── Combined: NAE only, all signals ──────────────────────────────────────────
print("Plotting combined NAE ROC...")
fig, ax = plt.subplots(figsize=(10, 8))
for (name, (_, s_nae)), color in zip(signal_results.items(), palette):
    y_true = np.concatenate([np.zeros(len(bg_nae_scores)), np.ones(len(s_nae))])
    fpr, tpr, _ = roc_curve(y_true, np.concatenate([bg_nae_scores, s_nae]))
    ax.plot(fpr, tpr, linewidth=2, color=color,
            label=f'{name} (AUC = {auc(fpr, tpr):.3f})')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.2, linewidth=1)
ax.set_xlabel('False Positive Rate (Zero Bias)')
ax.set_ylabel('True Positive Rate')
ax.set_title('NAE — All Signals')
ax.legend(fontsize=9, loc='lower right')
plt.tight_layout(pad=1.5)
save_fig(fig, 'roc_nae_all')


# ── Combined: AE vs NAE, all signals ─────────────────────────────────────────
print("Plotting combined AE vs NAE ROC...")
fig, ax = plt.subplots(figsize=(10, 8))
for (name, (s_ae, s_nae)), color in zip(signal_results.items(), palette):
    y_true_ae  = np.concatenate([np.zeros(len(bg_ae_scores)),  np.ones(len(s_ae))])
    y_true_nae = np.concatenate([np.zeros(len(bg_nae_scores)), np.ones(len(s_nae))])
    fpr_ae,  tpr_ae,  _ = roc_curve(y_true_ae,  np.concatenate([bg_ae_scores,  s_ae]))
    fpr_nae, tpr_nae, _ = roc_curve(y_true_nae, np.concatenate([bg_nae_scores, s_nae]))
    ax.plot(fpr_ae,  tpr_ae,  linestyle='--', linewidth=1.4, color=color, alpha=0.5)
    ax.plot(fpr_nae, tpr_nae, linestyle='-',  linewidth=2.0, color=color,
            label=name)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.2, linewidth=1)
ax.set_xlabel('False Positive Rate (Zero Bias)')
ax.set_ylabel('True Positive Rate')
ax.set_title('AE (dashed) vs NAE (solid) — All Signals')
ax.legend(fontsize=9, loc='lower right')
plt.tight_layout(pad=1.5)
save_fig(fig, 'roc_ae_vs_nae_all')

print("\nDone.")
