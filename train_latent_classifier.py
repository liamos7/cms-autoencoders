"""
Train XGBoost and MLP classifiers on CICADA latent space features,
then generate evaluation plots. Designed to run on a cluster node.
"""

import os
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["TF_NUM_INTEROP_THREADS"] = "4"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import gc
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for cluster use
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
import xgboost as xgb

import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(8)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ── Config ────────────────────────────────────────────────────────────────────

H5_DIR   = "/scratch/network/lo8603/thesis/fast-ad/data/h5_files"
OUT_DIR  = "/scratch/network/lo8603/thesis/plots/latent_classifier"
os.makedirs(OUT_DIR, exist_ok=True)

NAMES = [
    "glugluhtogg",
    "glugluhtotautau",
    "hto2longlivedto4b",
    "singleneutrino",
    "suep",
    "tt",
    "vbfhto2b",
    "vbfhtotautau",
    "zb",
    "zprimetotautau",
    "zz",
]

NPV_THRESHOLD = 10

# ── Load data ─────────────────────────────────────────────────────────────────

print("Loading HDF5 files...")
latent_spaces  = {}
student_scores = {}
npv            = {}

for name in NAMES:
    with h5py.File(f"{H5_DIR}/{name}.h5", "r") as f:
        latent_spaces[name]  = f["teacher_latent"][:]
        student_scores[name] = f["student_score"][:]
        npv[name]            = f["nPV"][:]
    print(f"  {name:<22}  latent={latent_spaces[name].shape}  scores={student_scores[name].shape}")

# ── Plot 1: student score distributions ──────────────────────────────────────

print("\nPlotting student score distributions...")
fig, ax = plt.subplots(figsize=(10, 6))
bins = np.linspace(
    min(s.min() for s in student_scores.values()),
    max(s.max() for s in student_scores.values()),
    80,
)
for label, scores in student_scores.items():
    ax.hist(scores, bins=bins, histtype="step", linewidth=1.5, label=label, density=True)
ax.set_xlabel("CICADA Student Score")
ax.set_ylabel("Density")
ax.set_title("Student Score Distributions by Event Type")
ax.legend(fontsize=8)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/student_score_distributions.png", dpi=150)
plt.close(fig)
print(f"  Saved student_score_distributions.png")

# ── Plot 2: nPV cut visualisation (ZeroBias) ─────────────────────────────────

print("Plotting nPV cut visualisation...")
from matplotlib.colors import LogNorm

npv_zb    = npv["zb"].astype(np.int32)
scores_zb = student_scores["zb"]
cut_pass  = npv_zb >= NPV_THRESHOLD
cut_fail  = ~cut_pass

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

bins = np.linspace(scores_zb.min(), scores_zb.max(), 80)
axes[0].hist(scores_zb[cut_fail], bins=bins, histtype="step", linewidth=1.5,
             color="tab:red",  density=True, label=f"nPV < {NPV_THRESHOLD}  (removed)")
axes[0].hist(scores_zb[cut_pass], bins=bins, histtype="step", linewidth=1.5,
             color="tab:blue", density=True, label=f"nPV ≥ {NPV_THRESHOLD}  (kept)")
axes[0].set_xlabel("CICADA Student Score")
axes[0].set_ylabel("Density")
axes[0].set_title("ZB score distribution: before vs. after nPV cut")
axes[0].legend()

hb = axes[1].hexbin(npv_zb, scores_zb, gridsize=60, cmap="viridis", mincnt=1, norm=LogNorm())
fig.colorbar(hb, ax=axes[1], label="Event count")
axes[1].axvline(NPV_THRESHOLD, color="tab:red", linewidth=1.5,
                linestyle="--", label=f"nPV = {NPV_THRESHOLD} cut")
axes[1].set_xlabel("nPV (number of primary vertices)")
axes[1].set_ylabel("CICADA Student Score")
axes[1].set_title("ZB: Student Score vs nPV")
axes[1].legend()

plt.suptitle(
    f"ZeroBias pileup cut: nPV ≥ {NPV_THRESHOLD}  "
    f"({cut_pass.sum():,} / {len(npv_zb):,} events kept,  {100*cut_pass.mean():.1f}%)",
    y=1.02,
)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/npv_cut_visualisation.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved npv_cut_visualisation.png")

# ── Data preparation ──────────────────────────────────────────────────────────

print("\nPreparing training data...")
class_names = NAMES
n_classes   = len(class_names)

min_len = min(len(v) for v in latent_spaces.values())
print(f"Min class size: {min_len:,} — truncating all classes to this.")

X = np.concatenate([latent_spaces[n][:min_len].astype(np.float32) for n in class_names], axis=0)
y = np.concatenate([np.full(min_len, i, dtype=np.int32) for i in range(n_classes)])
print(f"Total events: {len(y):,}  |  Latent dims: {X.shape[1]}")

del latent_spaces, student_scores, npv
gc.collect()

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
del X, y
gc.collect()

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)
del X_temp, y_temp
gc.collect()

print(f"Split → train: {len(y_train):,}, val: {len(y_val):,}, test: {len(y_test):,}")

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
X_test_sc  = scaler.transform(X_test)
del X_train, X_val, X_test
gc.collect()

class_weights_arr    = compute_class_weight("balanced", classes=np.arange(n_classes), y=y_train)
class_weight_dict    = dict(enumerate(class_weights_arr))
sample_weights_train = compute_sample_weight("balanced", y_train)

# ── XGBoost ───────────────────────────────────────────────────────────────────

print("\nTraining XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    device="cpu",
    nthread=8,
    eval_metric="mlogloss",
    early_stopping_rounds=20,
    random_state=42,
)
xgb_model.fit(
    X_train_sc, y_train,
    sample_weight=sample_weights_train,
    eval_set=[(X_val_sc, y_val)],
    verbose=50,
)

y_pred_xgb = xgb_model.predict(X_test_sc)
y_prob_xgb = xgb_model.predict_proba(X_test_sc)

print(f"\nBest iteration: {xgb_model.best_iteration}")
print("\nXGBoost – test-set classification report:")
print(classification_report(y_test, y_pred_xgb, target_names=class_names, digits=3))

# ── MLP ───────────────────────────────────────────────────────────────────────

print("\nTraining MLP...")
mlp = Sequential([
    Dense(256, activation="relu", input_shape=(X_train_sc.shape[1],)),
    Dropout(0.3),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(64,  activation="relu"),
    Dropout(0.2),
    Dense(n_classes, activation="softmax"),
], name="latent_classifier")

mlp.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
mlp.summary()

early_stop = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)

history = mlp.fit(
    X_train_sc, y_train,
    validation_data=(X_val_sc, y_val),
    epochs=300,
    batch_size=512,
    class_weight=class_weight_dict,
    callbacks=[early_stop],
    verbose=1,
)

y_prob_mlp = mlp.predict(X_test_sc)
y_pred_mlp = np.argmax(y_prob_mlp, axis=1)

print("\nMLP – test-set classification report:")
print(classification_report(y_test, y_pred_mlp, target_names=class_names, digits=3))

# ── Plot 3: MLP training curves ───────────────────────────────────────────────

print("\nPlotting MLP training curves...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history["loss"],     label="train")
axes[0].plot(history.history["val_loss"], label="val")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].set_title("Cross-entropy loss"); axes[0].legend()

axes[1].plot(history.history["accuracy"],     label="train")
axes[1].plot(history.history["val_accuracy"], label="val")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
axes[1].set_title("Accuracy"); axes[1].legend()

plt.suptitle("MLP training history", y=1.02)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/mlp_training_curves.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved mlp_training_curves.png")

# ── Plot 4: confusion matrices ────────────────────────────────────────────────

def plot_confusion_matrix(cm, class_names, title, ax):
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
    )
    ax.tick_params(axis="x", rotation=45)
    thresh = 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color=color, fontsize=8)

print("Plotting confusion matrices...")
cm_xgb = confusion_matrix(y_test, y_pred_xgb, normalize="true")
cm_mlp = confusion_matrix(y_test, y_pred_mlp, normalize="true")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
plot_confusion_matrix(cm_xgb, class_names, "XGBoost (normalised)", axes[0])
plot_confusion_matrix(cm_mlp, class_names, "MLP (normalised)",     axes[1])
plt.suptitle("Confusion matrices — fraction of true class predicted as each label")
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved confusion_matrices.png")

# ── Plot 5: per-class ROC curves ──────────────────────────────────────────────

print("Plotting ROC curves...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, y_prob, model_name in [
    (axes[0], y_prob_xgb, "XGBoost"),
    (axes[1], y_prob_mlp, "MLP"),
]:
    for k, name in enumerate(class_names):
        y_bin = (y_test == k).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_prob[:, k])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=1.5, label=f"{name}  (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"{model_name} — one-vs-rest ROC curves")
    ax.legend(fontsize=7)

plt.suptitle("Per-class ROC curves (one-vs-rest)")
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/roc_curves.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved roc_curves.png")

# ── Plot 6: XGBoost feature importances ──────────────────────────────────────

print("Plotting XGBoost feature importances...")
importances = xgb_model.feature_importances_
top_n   = 20
top_idx = np.argsort(importances)[-top_n:][::-1]

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(np.arange(top_n), importances[top_idx])
ax.set_xticks(np.arange(top_n))
ax.set_xticklabels([f"z{i}" for i in top_idx], rotation=45, ha="right")
ax.set_xlabel("Latent dimension")
ax.set_ylabel("Feature importance (XGBoost weight)")
ax.set_title(f"Top {top_n} most discriminating latent dimensions")
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/xgb_feature_importances.png", dpi=150)
plt.close(fig)
print(f"  Saved xgb_feature_importances.png")

# ── Plot 7: AUC bar chart (XGBoost vs MLP) ───────────────────────────────────

print("Plotting AUC bar chart...")
aucs = {"XGBoost": [], "MLP": []}
for k in range(n_classes):
    y_bin = (y_test == k).astype(int)
    for label, y_prob in [("XGBoost", y_prob_xgb), ("MLP", y_prob_mlp)]:
        aucs[label].append(auc(*roc_curve(y_bin, y_prob[:, k])[:2]))

x     = np.arange(n_classes)
width = 0.35

fig, ax = plt.subplots(figsize=(12, 5))
bars_xgb = ax.bar(x - width/2, aucs["XGBoost"], width, label="XGBoost", color="tab:blue")
bars_mlp = ax.bar(x + width/2, aucs["MLP"],     width, label="MLP",     color="tab:orange")

ax.set_xticks(x)
ax.set_xticklabels(class_names, rotation=45, ha="right")
ax.set_ylabel("One-vs-rest AUC")
ax.set_title("Per-class AUC — latent space features")
ax.set_ylim(0, 1.05)
ax.axhline(0.5, color="k", linestyle="--", linewidth=0.8, label="random")
ax.legend()

for bar in [*bars_xgb, *bars_mlp]:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)

plt.tight_layout()
fig.savefig(f"{OUT_DIR}/auc_bar_chart.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved auc_bar_chart.png")

print(f"\nAll plots saved to {OUT_DIR}/")
