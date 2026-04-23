"""
Fixes two problems with the previous pipeline:

1. OLD: read the first MAX_EVENTS rows from each ROOT file, then apply nPV cut.
       This biased the sample toward whatever data-taking period was written
       first. For ZB with 35.6M rows, the first 2M happened to be dominated by
       a specific high-pileup spike region, leaving the physically realistic
       nPV ~ 30-60 bulk essentially absent from the HDF5.

   NEW: uniformly sample a large index set across the full file, apply nPV cut,
       keep up to TARGET_EVENTS survivors. Seeded, reproducible.

2. OLD: loaded all branches into memory at once, which can be many GB.

   NEW: read branches in chunks; filter per chunk; discard before next chunk.

Outputs match the original HDF5 schema exactly, so downstream training and
analysis scripts work unchanged.
"""

import os
import gc
import glob
import argparse

import numpy as np
import awkward as ak
import uproot
import h5py
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import Model


# ─── Configuration ────────────────────────────────────────────────────────────
TEACHER_PATH = "/eos/user/l/ligerlac/cicada_data/models/work-in-progress/teacher-v1.0.0"
STUDENT_PATH = "/eos/user/l/ligerlac/cicada_data/models/work-in-progress/cicada-v2.2.0"

# How many events we want in the final HDF5 (after nPV cut). Script will
# sample enough raw rows to hit this target allowing for the expected ~91%
# nPV pass rate on ZB. For BSM files, pass rate is usually ~100% so the
# target is reached quickly.
TARGET_EVENTS = 8_000_000

# Chunk size for streaming reads. 500k rows × ~20 branches fits comfortably
# in <5 GB RAM for most branch types. Reduce if your node has <16 GB.
CHUNK_SIZE = 500_000

# nPV cut: drop events with fewer than this many reconstructed primary
# vertices. 10 is approximately what the published CICADA analyses use;
# it removes the genuinely-empty nPV ~ 0-1 spike.
NPV_THRESHOLD = 10

ET_REGIONS_SHAPE = (18, 14)

# Reproducibility
RNG_SEED = 42

# GPU memory cap (MB)
GPU_MEMORY_LIMIT = 8192


# ─── TensorFlow setup ─────────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=GPU_MEMORY_LIMIT)]
        )
    except RuntimeError as e:
        print(f"GPU config warning: {e}")


def make_dataset(arr, batch_size):
    def gen():
        for i in range(0, len(arr), batch_size):
            yield arr[i:i+batch_size]
    return tf.data.Dataset.from_generator(
        gen,
        output_signature=tf.TensorSpec(shape=(None,) + arr.shape[1:], dtype=tf.float32),
    )


def predict_in_chunks(model, arr, batch_size, infer_chunk=200_000):
    """Run model.predict in CPU-sized chunks to avoid GPU OOM on large arrays.

    model.predict() concatenates all batch outputs on the GPU before returning,
    which OOMs when the full array is ~8 GB. This function splits into chunks
    of `infer_chunk` rows, predicts each on GPU, and concatenates on CPU.
    """
    results = []
    for start in range(0, len(arr), infer_chunk):
        chunk = arr[start:start + infer_chunk]
        results.append(model.predict(make_dataset(chunk, batch_size), verbose=0))
        print(f"  infer chunk {start + len(chunk):,} / {len(arr):,}")
    return np.concatenate(results, axis=0)


# ─── Sampling + reading ───────────────────────────────────────────────────────
def uniform_sample_and_cut(input_root, target_events, rng):
    """
    Uniformly sample rows from the full ROOT file, apply nPV >= threshold,
    and stop when we have `target_events` survivors.

    Strategy:
      1. Get total row count.
      2. If total <= target_events / expected_pass_rate, read everything.
      3. Otherwise, draw a uniformly-random sorted index set sized to hit
         the target in expectation (with a safety multiplier).
      4. Stream those indices in chunks, apply nPV cut per chunk, accumulate.

    Returns dict of np arrays (one per branch) and the number of survivors.
    """
    print(f"Opening {input_root} ...")
    with uproot.open(input_root) as f:
        if "Events" not in f:
            raise KeyError(f"'Events' tree not found. Available: {list(f.keys())}")
        tree = f["Events"]
        n_total = tree.num_entries
        branches = [k for k in tree.keys()]
        print(f"  Total rows in file: {n_total:,}")
        print(f"  Branches: {len(branches)}")

        # Decide how many to sample. Assume a pessimistic 70% nPV pass rate
        # (actual is ~91% on ZB, ~100% on BSM, so this is a safe oversample).
        safety_factor = 1.5
        n_to_read = min(n_total, int(target_events / 0.70 * safety_factor))
        if n_to_read >= n_total:
            print(f"  Reading ALL {n_total:,} rows (smaller than sample target)")
            sample_indices = None   # signal: read everything
        else:
            print(f"  Uniformly sampling {n_to_read:,} of {n_total:,} "
                  f"({100*n_to_read/n_total:.1f}%)")
            sample_indices = np.sort(
                rng.choice(n_total, size=n_to_read, replace=False)
            ).astype(np.int64)

        # Accumulate per-branch arrays across chunks
        accum = {b: [] for b in branches}
        total_kept = 0

        if sample_indices is None:
            # Simple path: stream the full file in chunks
            iterator = _stream_full(tree, branches)
            total_to_process = tree.num_entries
        else:
            # Sampled path: stream chunks of the sample indices
            iterator = _stream_indices(tree, branches, sample_indices)
            total_to_process = len(sample_indices)

        # Process the FULL sample (no early stopping). Early-stopping on a
        # sorted sample introduces exactly the bias we're trying to fix:
        # you'd keep only the lower-indexed portion of the file.
        for chunk in iterator:
            npv = ak.to_numpy(chunk["nPV"])
            mask = npv >= NPV_THRESHOLD
            kept_here = int(mask.sum())
            if kept_here == 0:
                continue
            for b in branches:
                accum[b].append(ak.to_numpy(chunk[b])[mask])
            total_kept += kept_here
            print(f"    chunk kept {kept_here:,}  "
                  f"(running total {total_kept:,})")

        # Concatenate per-branch, then randomly subsample to target size.
        # Using the same rng ensures this is reproducible; shuffling avoids
        # any residual positional bias from chunk ordering.
        if total_kept > target_events:
            trim_idx = np.sort(
                rng.choice(total_kept, size=target_events, replace=False)
            )
            final_count = target_events
        else:
            trim_idx = np.arange(total_kept)
            final_count = total_kept

        result = {}
        for b in branches:
            if accum[b]:
                arr = np.concatenate(accum[b], axis=0)
                result[b] = arr[trim_idx]

        pass_rate = 100 * total_kept / total_to_process
        print(f"  Final kept: {final_count:,}  "
              f"(survived nPV cut: {total_kept:,} of {total_to_process:,} = {pass_rate:.1f}%)")
        return result, final_count


def _stream_full(tree, branches):
    """Yield chunks covering the full tree."""
    for start in range(0, tree.num_entries, CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, tree.num_entries)
        yield tree.arrays(branches, entry_start=start, entry_stop=stop, library="ak")


def _stream_indices(tree, branches, indices):
    """
    Yield chunks of rows at the given (sorted) indices.

    Strategy: group indices into dense runs per CHUNK_SIZE window so that
    each uproot read is contiguous (fast). Drop rows not in the sample set.
    """
    n_total = tree.num_entries
    # Partition indices into [start, start+CHUNK_SIZE) windows
    windows = {}
    for idx in indices:
        key = (idx // CHUNK_SIZE) * CHUNK_SIZE
        windows.setdefault(key, []).append(idx)

    for start in sorted(windows.keys()):
        stop = min(start + CHUNK_SIZE, n_total)
        local_idx = np.array(windows[start], dtype=np.int64) - start
        chunk = tree.arrays(branches, entry_start=start, entry_stop=stop, library="ak")
        yield chunk[local_idx]


# ─── CICADA decoration ────────────────────────────────────────────────────────
def run_cicada(data):
    """Decorate data dict with teacher_score, teacher_latent, student_score, total_et."""
    et_regions = data["et_regions"].reshape(-1, *ET_REGIONS_SHAPE).astype(np.float32)

    # Teacher
    print("Loading teacher ...")
    teacher_model = load_model(TEACHER_PATH)
    encoder = Model(
        inputs=teacher_model.input,
        outputs=teacher_model.get_layer("teacher_latent").output,
    )

    print("Running teacher inference ...")
    teacher_preds = predict_in_chunks(teacher_model, et_regions, batch_size=32).squeeze(-1)
    mse = np.mean((et_regions - teacher_preds) ** 2, axis=(1, 2))
    data["teacher_score"] = np.clip(32 * np.log(mse), 0, 128).astype(np.float32)

    print("Running encoder inference ...")
    data["teacher_latent"] = predict_in_chunks(encoder, et_regions, batch_size=32).astype(np.float32)

    del teacher_model, encoder
    tf.keras.backend.clear_session()
    gc.collect()

    # Student
    print("Loading student ...")
    student_model = load_model(STUDENT_PATH)
    X_flat = et_regions.reshape(len(et_regions), -1)
    print("Running student inference ...")
    data["student_score"] = predict_in_chunks(student_model, X_flat, batch_size=1024).squeeze().astype(np.float32)

    del student_model
    tf.keras.backend.clear_session()
    gc.collect()

    data["total_et"] = et_regions.sum(axis=(1, 2)).astype(np.float32)
    return data


# ─── HDF5 output ──────────────────────────────────────────────────────────────
def write_hdf5(data, output_h5, n_kept):
    print(f"Writing {output_h5} ...")
    out_dir = os.path.dirname(output_h5)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with h5py.File(output_h5, "w") as h5f:
        for name, arr in data.items():
            if name == "et_regions" and arr.ndim == 2 and arr.shape[1] == 252:
                arr = arr.reshape(-1, *ET_REGIONS_SHAPE)
                chunks = (min(1000, n_kept), *ET_REGIONS_SHAPE)
            else:
                chunks = True
            h5f.create_dataset(
                name, data=arr, compression="gzip",
                compression_opts=4, chunks=chunks,
            )
            print(f"  wrote {name}: {arr.shape}  dtype={arr.dtype}")

        # Record provenance for reproducibility
        h5f.attrs["rng_seed"] = RNG_SEED
        h5f.attrs["npv_threshold"] = NPV_THRESHOLD
        h5f.attrs["target_events"] = TARGET_EVENTS
        h5f.attrs["actual_events"] = n_kept
        h5f.attrs["sampling"] = "uniform_random_across_full_file"

    print(f"Done. {n_kept:,} events → {output_h5}")


# ─── Per-file pipeline ────────────────────────────────────────────────────────
def process_file(root_path, out_path, rng):
    print(f"\n{'=' * 70}")
    print(f"Processing: {os.path.basename(root_path)} → {out_path}")
    print(f"{'=' * 70}")
    data, n_kept = uniform_sample_and_cut(root_path, TARGET_EVENTS, rng)
    if n_kept == 0:
        print("  WARNING: zero events survived cut, skipping write")
        return
    data = run_cicada(data)
    write_hdf5(data, out_path, n_kept)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir", default="Data/raw_root",
        help="Directory containing raw ROOT files",
    )
    parser.add_argument(
        "--out-dir", default="Data/hdf5_files",
        help="Output directory for HDF5 files",
    )
    parser.add_argument(
        "--only", default=None,
        help="Process only this one file (basename, e.g. 'zb.root')",
    )
    args = parser.parse_args()

    root_files = sorted(glob.glob(os.path.join(args.raw_dir, "*.root")))
    if args.only:
        root_files = [p for p in root_files if os.path.basename(p) == args.only]
    if not root_files:
        print(f"No ROOT files found in {args.raw_dir}")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Found {len(root_files)} ROOT file(s) to process")
    print(f"Target events per file: {TARGET_EVENTS:,}")
    print(f"nPV threshold: {NPV_THRESHOLD}")
    print(f"RNG seed: {RNG_SEED}")

    # One shared RNG across all files for reproducibility; each file gets
    # a deterministic but different sample.
    rng = np.random.default_rng(RNG_SEED)

    for root_path in root_files:
        stem = os.path.splitext(os.path.basename(root_path))[0]
        out_path = os.path.join(args.out_dir, stem + ".h5")
        process_file(root_path, out_path, rng)

    print("\nAll files processed.")


if __name__ == "__main__":
    main()