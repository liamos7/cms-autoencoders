import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path

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

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 150,
})

CMAP = "inferno"


MEAN_SAMPLE = 10_000


def pick_event(h5path):
    """Return the event whose spatial ET pattern is closest to the process mean image."""
    with h5py.File(h5path, "r") as f:
        n = f["et_regions"].shape[0]
        # Subsample for mean computation
        sample_n = min(MEAN_SAMPLE, n)
        rng = np.random.default_rng(42)
        sample_idx = np.sort(rng.choice(n, sample_n, replace=False))
        sample = f["et_regions"][list(sample_idx)].astype(np.float32)  # (S, 18, 14)
        mean_image = sample.mean(axis=0)  # (18, 14)

        # Find event in the full dataset closest to mean image by L2 distance.
        # To avoid loading all N events, score in chunks.
        best_idx, best_dist = 0, np.inf
        CHUNK = 50_000
        for start in range(0, n, CHUNK):
            end = min(start + CHUNK, n)
            chunk = f["et_regions"][start:end].astype(np.float32)
            dists = np.mean((chunk - mean_image) ** 2, axis=(1, 2))
            local_best = int(np.argmin(dists))
            if dists[local_best] < best_dist:
                best_dist = dists[local_best]
                best_idx = start + local_best

        event = f["et_regions"][best_idx].astype(np.float32)
    return event


def plot_et_regions(h5_files, output_path):
    n_files = len(h5_files)
    n_cols = int(np.ceil(np.sqrt(n_files)))
    n_rows = int(np.ceil(n_files / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.8 * n_cols, 3.5 * n_rows),
        constrained_layout=True,
    )
    axes = np.array(axes).flatten()

    events = {Path(p).stem: pick_event(p) for p in h5_files}

    # Shared scale across all plots: clip empty cells to vmin so log(0) is avoided
    VMIN = 0.5
    vmax = max(float(e.max()) for e in events.values())
    norm = mcolors.LogNorm(vmin=VMIN, vmax=max(vmax, 1))

    for ax, path in zip(axes, h5_files):
        stem = Path(path).stem
        event = events[stem]   # (18, 14)
        label = LABEL_MAP.get(stem, stem)

        # Clip zeros to vmin so LogNorm doesn't blow up on empty cells
        display = np.where(event > 0, event, VMIN)

        im = ax.imshow(
            display.T,         # (14, 18): phi on y-axis, eta on x-axis
            origin="lower",
            aspect="auto",
            cmap=CMAP,
            norm=norm,
            interpolation="nearest",
        )

        ax.set_title(label, pad=5)
        ax.set_xlabel(r"$i\eta$", labelpad=2)
        ax.set_ylabel(r"$i\phi$", labelpad=2)

        ax.set_xticks(range(0, 18, 2))
        ax.set_yticks(range(0, 14, 2))
        ax.set_xticklabels(range(1, 19, 2))
        ax.set_yticklabels(range(1, 15, 2))

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = fig.colorbar(im, cax=cax)
        nice_ticks = [t for t in [1, 10, 50, 200, 500, 1000] if VMIN <= t <= vmax]
        if nice_ticks:
            cb.set_ticks(nice_ticks)
            cb.set_ticklabels([str(t) for t in nice_ticks])
        cb.set_label("ET [GeV]", fontsize=7, labelpad=4)
        cb.ax.tick_params(labelsize=6)

    for ax in axes[n_files:]:
        ax.set_visible(False)

    fig.suptitle(r"ET Regions", fontsize=14)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    h5_dir = Path("h5_files")
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)

    h5_files = sorted(h5_dir.glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in {h5_dir}")

    plot_et_regions(h5_files, output_dir / "et_regions_sample.png")
