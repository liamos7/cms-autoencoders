# runs on CERN EOS


import uproot
import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

CUT = 10

f = uproot.open("Data/raw_root/zb.root")
npv_root = f["Events"]["nPV"].array(library="np")

with h5py.File("Data/hdf5_files/zb.h5", "r") as f:
    npv_h5 = f["nPV"][:]

datasets = [
    (npv_root, "raw_root/zb.root"),
    (npv_h5,   "hdf5_files/zb.h5"),
]

fmt = mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}k")

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=False)
fig.suptitle("NPV Distribution â€” Zero Bias", fontsize=15, y=1.01)

for ax, (npv, label) in zip(axes, datasets):
    frac_cut = (npv <= CUT).sum() / len(npv) * 100

    bins = np.arange(npv.min(), npv.max() + 2) - 0.5
    counts, edges, patches = ax.hist(npv, bins=bins, color="#4C72B0", edgecolor="white", linewidth=0.3, alpha=0.85)

    for patch, left in zip(patches, edges[:-1]):
        if left <= CUT:
            patch.set_facecolor("#C44E52")
            patch.set_alpha(0.75)

    ax.axvline(CUT, color="#C44E52", linestyle="--", linewidth=1.8,
               label=f"NPV = {CUT} cut  ({frac_cut:.2f}% removed)")

    ax.set_xlabel("Number of Primary Vertices (NPV)", fontsize=12)
    ax.set_ylabel("Events", fontsize=12)
    ax.set_title(label, fontsize=12, pad=8)
    ax.yaxis.set_major_formatter(fmt)
    ax.legend(fontsize=10.5, frameon=False)

    stats_text = (
        f"Entries: {len(npv):,}\n"
        f"Mean: {npv.mean():.1f}\n"
        f"Std:  {npv.std():.1f}"
    )
    ax.text(0.97, 0.95, stats_text, transform=ax.transAxes, ha="right", va="top",
            fontsize=10, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.9))

plt.tight_layout()
plt.savefig("plots/npv_distribution.png", dpi=150, bbox_inches="tight")
print("Saved plots/npv_distribution.png")
for npv, label in datasets:
    frac = (npv <= CUT).sum() / len(npv) * 100
    print(f"  {label}: {len(npv):,} entries, {frac:.2f}% removed by NPV > {CUT} cut")
plt.show()