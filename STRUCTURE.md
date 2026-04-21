# Thesis Project Structure

This is an anomaly detection project for particle physics, focused on training a
Normalized Autoencoder (NAE) for Beyond-Standard-Model (BSM) physics detection
at the LHC/CMS CICADA L1 trigger system.

---

## Top-Level Files

| File | Description |
|------|-------------|
| `correlations.py` | t-SNE visualization of the CICADA latent space; correlation heatmaps of observables (nPV, ET, η, HT) across signal/background classes. Outputs to `plots/tsne/` and `plots/correlations/`. |
| `lasso_analysis.py` | LASSO regression to identify which latent dimensions predict observables (teacher_score, total_et, nPV, first_jet_eta, ht). Computes regularization paths and R². Outputs to `plots/lasso/`. Takes ~2h on 500k events. |
| `train_et_regions_classifier.py` | Trains XGBoost and MLP classifiers on raw 18×14 calorimeter grid (252 flattened features) to classify signal vs background. Comparison baseline. |
| `train_latent_classifier.py` | Trains XGBoost and MLP classifiers on the 80-dim latent space features from the CICADA teacher encoder. Outputs to `plots/latent_classifier/`. |
| `plot_training.py` | Reads TensorBoard event files and generates clean training curve figures (positive/negative energy, AUC vs epoch). Outputs to `plots/training/`. |
| `pkl_cpu.py` | Utility to convert GPU model checkpoints to CPU-loadable format. |
| `test_imports.py` | Sanity check: imports torch, numpy, sklearn, etc. and prints versions/CUDA availability. |

---

## `fast-ad/` — Main ML Project

### `fast-ad/fastad/` — Core ML Library

**`models/`**

| File | Description |
|------|-------------|
| `modules.py` | Low-level building blocks: `SimpleEncoder` (conv layers → latent dim), `SimpleDecoder` (conv transpose → 18×14 image), `CicadaDecoder` (with sigmoid output), Gaussian/Laplace distribution classes. |
| `teachers.py` | Main model implementations: `AE` (standard autoencoder, Phase 1) and `NAEWithEnergyTraining` (Phase 2 energy-based model). Implements contrastive divergence loss with Langevin Monte Carlo sampling. Energy is defined as reconstruction error. Contains 6 documented bug fixes. |
| `students.py` | Lightweight student models (`StudentA`, `StudentB`) for distillation experiments. Not core to thesis. |
| `nae.py` | Simplified NAE integration layer (minimal). |
| `__init__.py` | Factory functions: `get_teacher_model()`, `get_cicada_ae()`, `get_cicada_nae_with_energy()`. |

**Other modules**

| File | Description |
|------|-------------|
| `datasets.py` | CICADA dataset loader with 3-way (80/10/10) stratified split. Label 0 = ZB background; labels 1–10 = signal processes (ggH→γγ, ggH→ττ, SUEP, tt, ZZ, etc.). Log-normalizes 18×14 calorimeter images. |
| `trainers.py` | `BaseTrainer`: iterative training loop. Tracks best model by loss (Phase 1) or AUC (Phase 2). Logs metrics via TensorBoard. |
| `loggers.py` | `BaseLogger` for TensorBoard integration. Accumulates scalars and images per epoch. |
| `utils.py` | ROC-AUC computation, argument parsing, averaging meters. |

### `fast-ad/` Root Scripts

| File | Description |
|------|-------------|
| `train-teacher.py` | Entry point for Phase 1 (AE) and Phase 2 (NAE) training. Args: `--model {AE, NAEWithEnergyTraining}`, `--latent-dim`, `--load-pretrained-path`, `--epochs`, `-o` (output dir). |
| `ae_vs_nae_rocs.py` | Loads best AE and best NAE checkpoints; scores test split for each signal class; produces overlaid ROC curves and AUC table. Uses stratified 80/10/10 split. |
| `eval_latent_dim_rocs.py` | Iterates over `outputs/latent_dim_variation/`; computes ROC/AUC with 95% bootstrap CIs for each signal vs ZB at each latent dimension; produces AUC-vs-dim heatmap. |
| `CLAUDE.md` | Project documentation: training sequence, 6 code fixes (sigmoid decoder, energy regularization, gradient normalization, temperature decoupling, NaN handling, AUC-based selection), autoresearch sweep parameters. |

### `fast-ad/data/` — Dataset Preparation & Visualization

| File/Dir | Description |
|----------|-------------|
| `h5_files/` | 11 HDF5 files (~8.9 GB total). Each contains: `et_regions` (raw 18×14 images), `teacher_latent` (80-dim encodings), `teacher_score`, `total_et`, `nPV`, `first_jet_eta`, `ht`. Files: `zb.h5` (3.2 GB background), `glugluhtotautau.h5`, `glugluhtogg.h5`, `singleneutrino.h5`, `suep.h5`, `tt.h5`, `vbfhto2b.h5`, `vbfhtotautau.h5`, `zprimetotautau.h5`, `zz.h5`. |
| `observable_plotter.py` | Plots distributions of observables (nPV, total_et, first_jet_eta, ht) per class. |
| `et_regions_plotter.py` | Plots raw calorimeter grid distributions per class. |
| `pileup_correlation_plotter.py` | Investigates correlations between pileup (nPV) and reconstruction energy. |
| `teacher_roc.py` | ROC curves for the CICADA baseline ("teacher") model. |
| `plots/` | Output directory for the above visualizations. |

### `fast-ad/outputs/` — Model Checkpoints & Results (gitignored)

| Directory | Description |
|-----------|-------------|
| `ae_phase1_sigmoid_dim20/` | Phase 1 autoencoder (reconstruction-only, latent dim=20, sigmoid decoder). Used as pretrain starting point for Phase 2. |
| `nae_phase2_fixed_dim20/` | NAE Phase 2 with all 6 bug fixes applied. |
| `nae_phase2_tuned_dim20/` | Hand-tuned NAE Phase 2 hyperparameters (baseline reference). |
| `nae_phase2_tuned_dim80/` | NAE attempted in 80-dim spherical latent space. |
| `nae_mc_upper_bound_dim20/` | Oracle experiment: replace Langevin sampling with real background samples (ZB + SingleNeutrino). Establishes upper bound on contrastive objective. |
| `latent_dim_variation/` | Sweep of AE models at latent dims 3, 5, 10, 20, 32, 64, 80. Each subdir: `ae_zb_dim{N}/model_best.pkl`. |

Each checkpoint directory typically contains:
- `model_best.pkl` — PyTorch state dict (best by validation AUC or loss).
- `metrics.json` — Final training metrics.
- `events.out.tfevents.*` — TensorBoard logs.

### `fast-ad/nae-autoresearch/` — Autonomous Hyperparameter Search

An autonomous research agent that iteratively modifies `train.py` to maximize anomaly detection AUC.

| File | Description |
|------|-------------|
| `train.py` | **The modifiable file.** Contains all NAE Phase 2 hyperparameters (GAMMA, NEG_LAMBDA, Z_STEPS, Z_STEP_SIZE, X_STEPS, X_NOISE_STD, etc.), Langevin sampling loops, replay buffer, loss function, and architecture. The agent modifies only this file. |
| `evaluate.py` | **Read-only evaluation harness.** Runs `train.py`, reads `metrics.json`, computes score = best_val_auc × stability_multiplier. Collapsed runs score 0.0. |
| `program.md` | Agent instructions: metric definition, failure modes, 12 hyperparameter categories to explore, research strategy, known instabilities. |
| `run_autoresearch.sh` | Slurm job script: submits evaluation harness to GPU, keeps node alive for interactive agent. |
| `autoresearch_logs/` | Agent experiment history (JSON logs per run). |

### `fast-ad/plots/` — Generated Visualizations

| Subdirectory | Content |
|--------------|---------|
| `rocs_test/` | AE vs NAE ROC curves on held-out test split. |
| `training/` | Training curves (loss, AUC over epochs) from TensorBoard logs. |

---

## `slurm_scripts/` — HPC Job Submission

All scripts run on the adroit cluster (Slurm), load `anaconda3/2024.10` + `cudatoolkit/12.6`, and activate the conda env at `conda/envs/myenv`.

| Script | Description |
|--------|-------------|
| `run_training.sh` | Phase 1 AE training (24h walltime, 1 GPU, 64 GB RAM). |
| `run_classifiers.sh` | Runs both `train_et_regions_classifier.py` and `train_latent_classifier.py` sequentially (4h, GPU). |
| `run_lasso.sh` | Runs `lasso_analysis.py` (CPU-only, 4 cores, 16 GB RAM). |
| `eval_latent_dim_rocs.sh` | Evaluates ROC curves for all latent dim variations. |
| `run_ae_vs_nae_rocs.sh` | Generates AE vs NAE comparison ROC plots. |
| `train_latent_dim_sweep.sh` | Trains AE at multiple latent dimensions (parameter sweep). |
| `slurm_test.sh` | Minimal test to verify environment. |

---

## `plots/` — Root-Level Generated Plots

| Subdirectory | Content |
|--------------|---------|
| `lasso/` | LASSO regularization paths, active sets, R² per observable. |
| `correlations/` | Observable correlation heatmaps across classes. |
| `et_regions_classifier/` | Confusion matrices, ROC curves, feature importances (et_regions classifiers). |
| `latent_classifier/` | Same for latent-space classifiers. |
| `tsne/` | t-SNE embeddings of latent space. |
| `training/` | Training curve figures. |

---

## `logs/` — Slurm Job Logs (gitignored)

Stdout/stderr from all submitted jobs. Naming: `train_*.out/err`, `classifiers_*.out/err`, `lasso_*.out/err`, `rocs_*.out/err`, etc.

---

## Key Concepts

**Two-Phase Training**:
1. **Phase 1 (AE)**: Reconstruction-only autoencoder trained on ZB background. Encoder learns a compressed representation.
2. **Phase 2 (NAE)**: Energy-based contrastive training. Energy = reconstruction error. Langevin Monte Carlo samples off-manifold negatives. Contrastive divergence pushes background energy low and negative-sample energy high.

**Data**: 18×14 calorimeter trigger tower images from CMS CICADA. Background = Zero Bias (ZB) pileup events. Signals = ggH→γγ, ggH→ττ, SUEP, tt, ZZ, VBF H→bb, VBF H→ττ, Z→ττ, Z'.

**Environment**: Python 3.10, PyTorch, scikit-learn, XGBoost, h5py, matplotlib, mplhep, TensorBoard. GPU jobs on adroit cluster.
