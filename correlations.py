import os
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
from sklearn.manifold import TSNE

hep.style.use("CMS")

H5_DIR = "/scratch/network/lo8603/thesis/fast-ad/data/h5_files"
TSNE_DIR = "plots/tsne"
CORR_DIR = "plots/correlations"

os.makedirs(TSNE_DIR, exist_ok=True)
os.makedirs(CORR_DIR, exist_ok=True)

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

SAMPLES = [
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

CORR_SAMPLES = ["glugluhtogg"] + SAMPLES
ALL_SAMPLES  = ["glugluhtogg"] + SAMPLES


def plot_latent_tsne_with_observables(name, nmax=5000):
    with h5py.File(f"{H5_DIR}/{name}.h5", "r") as f:
        latent_space   = f["teacher_latent"][:nmax]
        student_scores = f["student_score"][:nmax]
        energy         = f["total_et"][:nmax]
        pileup         = f["nPV"][:nmax].astype(np.float32)
        first_jet_et   = f["first_jet_et"][:nmax]
        first_jet_eta  = f["first_jet_eta"][:nmax]
        ht             = f["ht"][:nmax]

    print(f"  Running t-SNE for {name}...")
    tsne = TSNE(n_components=2, perplexity=30, learning_rate="auto", init="pca", random_state=42)
    latent_2d = tsne.fit_transform(latent_space)

    panels = [
        (energy,         "Total Energy",  "viridis"),
        (student_scores, "Student Score", "plasma"),
        (pileup,         "Pileup (nPV)",  "inferno"),
        (first_jet_et,   "First Jet ET",  "magma"),
        (ht,             "HT",            "cividis"),
        (first_jet_eta,  "First Jet Eta", "coolwarm"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9),
                             gridspec_kw={"hspace": 0.35, "wspace": 0.05})
    fig.suptitle(name, fontsize=15, y=1.01)

    for ax, (values, label, cmap) in zip(axes.flat, panels):
        sc = ax.scatter(latent_2d[:, 0], latent_2d[:, 1], c=values, cmap=cmap, s=2, alpha=0.6)
        cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
        cb.set_label(label, fontsize=11)
        cb.ax.tick_params(labelsize=9)
        ax.set_title(label, fontsize=12, pad=4)
        ax.set_xlabel("t-SNE 1", fontsize=11)
        ax.set_ylabel("t-SNE 2", fontsize=11)
        ax.tick_params(labelsize=9)

    out = os.path.join(TSNE_DIR, f"tsne_{name}.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


def plot_latent_correlations(name, nmax=5000):
    with h5py.File(f"{H5_DIR}/{name}.h5", "r") as f:
        latents        = f["teacher_latent"][:nmax]
        student_scores = f["student_score"][:nmax]
        energy         = f["total_et"][:nmax]
        pileup         = f["nPV"][:nmax].astype(np.float32)
        first_jet_et   = f["first_jet_et"][:nmax]
        first_jet_eta  = f["first_jet_eta"][:nmax]
        ht             = f["ht"][:nmax]

    observables = [
        (energy,         "Energy"),
        (student_scores, "Student Score"),
        (pileup,         "Pileup (nPV)"),
        (first_jet_et,   "First Jet ET"),
        (ht,             "HT"),
        (first_jet_eta,  "First Jet Eta"),
    ]

    n_latent = latents.shape[1]
    xs = np.arange(n_latent)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7),
                             gridspec_kw={"hspace": 0.45, "wspace": 0.35})
    fig.suptitle(f"Latent–observable correlations: {name}", fontsize=13, y=1.02)

    for ax, (obs, label) in zip(axes.flat, observables):
        corrs = np.array([np.corrcoef(obs, latents[:, i])[0, 1] for i in xs])
        ax.scatter(xs, corrs, s=8, linewidths=0)
        ax.axhline(0, color="gray", lw=0.6, ls="--")
        ax.set_title(label, fontsize=12, pad=3)
        ax.set_xlabel("Latent index", fontsize=10)
        ax.set_ylabel("Pearson r", fontsize=10)
        ax.tick_params(labelsize=9)

    # hide x-label on top row to reduce clutter
    for ax in axes[0]:
        ax.set_xlabel("")

    out = os.path.join(CORR_DIR, f"correlations_{name}.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


def plot_combined_tsne(samples=ALL_SAMPLES, nmax=2000):
    """Run t-SNE on teacher_latent pooled across all datasets, colour by type."""
    all_latents = []
    all_labels  = []

    for name in samples:
        path = f"{H5_DIR}/{name}.h5"
        if not os.path.exists(path):
            print(f"  Skipping {name}: file not found")
            continue
        with h5py.File(path, "r") as f:
            latent = f["teacher_latent"][:nmax]
        all_latents.append(latent)
        all_labels.extend([name] * len(latent))
        print(f"  Loaded {name}: {len(latent):,} events")

    latents_arr = np.concatenate(all_latents, axis=0)
    labels_arr  = np.array(all_labels)

    print(f"  Running t-SNE on {len(latents_arr):,} events × {latents_arr.shape[1]} dims...")
    tsne = TSNE(n_components=2, perplexity=40, learning_rate="auto",
                init="pca", random_state=42, n_jobs=-1)
    coords = tsne.fit_transform(latents_arr)

    fig, ax = plt.subplots(figsize=(10, 8))

    unique_names = [s for s in samples if s in set(labels_arr)]
    for i, name in enumerate(unique_names):
        mask = labels_arr == name
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=3, alpha=0.4,
            color=COLORS[i % len(COLORS)],
            label=LABEL_MAP.get(name, name),
            rasterized=True,
        )

    ax.set_xlabel("t-SNE 1", fontsize=12)
    ax.set_ylabel("t-SNE 2", fontsize=12)
    ax.set_title("t-SNE of teacher latent space (all datasets)", fontsize=13)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
        markerscale=4,
        framealpha=0.9,
        edgecolor="0.7",
        fontsize=10,
    )
    ax.tick_params(labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = os.path.join(TSNE_DIR, "tsne_combined.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


if __name__ == "__main__":
    print("=== Combined t-SNE ===")
    plot_combined_tsne()

    print("\n=== Per-sample t-SNE plots ===")
    for name in SAMPLES:
        print(f"Processing {name}...")
        plot_latent_tsne_with_observables(name)

    print("\n=== Correlation plots ===")
    for name in CORR_SAMPLES:
        print(f"Processing {name}...")
        plot_latent_correlations(name)

    print("\nDone.")
