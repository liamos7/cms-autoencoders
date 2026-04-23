# Thesis Project Structure

This is an anomaly detection project for particle physics, focused on training a
Normalized Autoencoder (NAE) for Beyond-Standard-Model (BSM) physics detection
at the LHC/CMS CICADA L1 trigger system.

---

## Top-Level Files

| File | Description |
|------|-------------|
| `correlations.py` | t-SNE visualization of the CICADA latent space; correlation heatmaps of observables  across signal/background classes. Outputs to `plots/tsne/` and `plots/correlations/`. |
| `lasso_analysis.py` | LASSO regression to identify which latent dimensions predict observables (teacher_score, total_et, nPV, first_jet_eta, ht). Computes regularization paths and R^2. Outputs to `plots/lasso/`.  |
| `train_et_regions_classifier.py` | Trains XGBoost and MLP classifiers on raw 18x14 calorimeter grid (252 flattened features) to classify signal vs background. Comparison baseline. |
| `train_latent_classifier.py` | Trains XGBoost and MLP classifiers on the 80-dim latent space features from the CICADA teacher encoder. Outputs to `plots/latent_classifier/`. |
| `plot_training.py` | Reads TensorBoard event files and generates clean training curve figures (positive/negative energy, AUC vs epoch). Outputs to `plots/training/`. |
| `pkl_cpu.py` | Utility to convert GPU model checkpoints to CPU-loadable format. |
| `test_imports.py` | Sanity check: imports torch, numpy, sklearn, etc. and prints versions/CUDA availability. |
| `requirements_backup.txt` | Snapshot of pip-installed packages in `myenv` (for reproducibility reference). |

---

## `fast-ad/` — Main ML Project

### `fast-ad/fastad/` — Core ML Library

**`models/`**

| File | Description |
|------|-------------|
| `modules.py` | Low-level building blocks: `SimpleEncoder` (conv layers to latent dim), `SimpleDecoder` (conv transpose to 18x14 image), `CicadaDecoder` (with sigmoid output), Gaussian/Laplace distribution classes. |
| `teachers.py` | Main model implementations: `AE` (standard autoencoder, Phase 1) and `NAEWithEnergyTraining` (Phase 2 energy-based model). Implements contrastive divergence loss with Langevin Monte Carlo sampling. Energy is defined as reconstruction error. Contains documented bug fixes. |
| `students.py` | Lightweight student models (`StudentA`, `StudentB`) for distillation experiments. Not core to thesis. |
| `__init__.py` | Factory functions: `get_teacher_model()`, `get_cicada_ae()`, `get_cicada_nae_with_energy()`. |

**Other modules**

| File | Description |
|------|-------------|
| `datasets.py` | CICADA dataset loader with 3-way (80/10/10) stratified split. Label 0 = ZB background; labels 1–10 = signal processes. Log-normalizes 18x14 calorimeter images. |
| `trainers.py` | `BaseTrainer`: iterative training loop. Tracks best model by loss (Phase 1) or AUC (Phase 2). Logs metrics via TensorBoard. |
| `loggers.py` | `BaseLogger` for TensorBoard integration. Accumulates scalars and images per epoch. |
| `utils.py` | ROC-AUC computation, argument parsing, averaging meters. |

### `fast-ad/` Root Scripts

| File | Description |
|------|-------------|
| `train-teacher.py` | Entry point for Phase 1 (AE) and Phase 2 (NAE) training. Args: `--model {AE, NAEWithEnergyTraining}`, `--latent-dim`, `--load-pretrained-path`, `--epochs`, `-o` (output dir). |
| `ae_vs_nae_rocs.py` | Loads best AE and best NAE checkpoints; scores test split for each signal class; produces overlaid ROC curves and AUC table. Uses stratified 80/10/10 split. |
| `eval_latent_dim_rocs.py` | Iterates over `outputs/latent_dim_variation/`; computes ROC/AUC with 95% bootstrap CIs for each signal vs ZB at each latent dimension; produces per-signal ROC overlays and mean-AUC-vs-dim summary. |
| `nae_mc_oracle_rocs.py` | Oracle/upper-bound experiment: scores the MC-oracle NAE on all signal classes; produces ROC curves and oracle-vs-teacher comparison plots. |
| `CLAUDE.md` | Project documentation: training sequence, 6 code fixes (sigmoid decoder, energy regularization, gradient normalization, temperature decoupling, NaN handling, AUC-based selection), autoresearch sweep parameters. |

### `fast-ad/data/` — Dataset Preparation & Visualization

| File/Dir | Description |
|----------|-------------|
| `h5_files/` | 11 HDF5 files (gitignored). Each contains: `et_regions` (raw 18x14 images), `teacher_latent` (80-dim encodings), `teacher_score`, `total_et`, `nPV`, `first_jet_eta`, `ht`. Files: `zb.h5`, `glugluhtotautau.h5`, `glugluhtogg.h5`, `singleneutrino.h5`, `suep.h5`, `tt.h5`, `vbfhto2b.h5`, `vbfhtotautau.h5`, `zprimetotautau.h5`, `zz.h5`. |
| `skim-inputs-mp.py` | Skims CICADANtuples ROOT files on CERN EOS; parallelized. Runs on EOS — configure paths before use. |
| `process_to_hdf5.py` | Converts skimmed ROOT files to HDF5 format, applying nPV cuts properly (samples first then cuts, to avoid run-period bias). Fixes the biased-ordering problem of the previous pipeline. |
| `check_sampling_npv.py` | Diagnostic: verifies nPV distribution of sampled events matches expectation after the corrected sampling strategy. |
| `observable_plotter.py` | Plots distributions of observables (nPV, total_et, first_jet_eta, ht) per class. |
| `et_regions_plotter.py` | Plots raw calorimeter grid distributions per class. |
| `pileup_correlation_plotter.py` | Investigates correlations between pileup (nPV) and reconstruction energy. |
| `teacher_roc.py` | ROC curves for the CICADA baseline ("teacher") model. |
| `old_zb.h5` | Previous ZB HDF5 produced with the biased sampling pipeline (kept for comparison, gitignored). |
| `plots/` | Output directory for the above visualizations; includes `zb_eos_npv_distribution.png` (nPV diagnostic). |

### `fast-ad/outputs/` — Model Checkpoints & Result

| Directory | Description |
|-----------|-------------|

| `nae_phase2_dim20_zb_1/` | NAE with LMC, 20-dimensional latent space with n_z=30, λ_z=0.005, T=1.0, γ=0.01. |
| `nae_phase2_dim20_zb_2/` | NAE with LMC, 20-dimensional latent space with n_z=60, λ_z=0.01, T=0.5, γ=0.005. |
| `nae_mc_oracle_dim20_zb/` | Oracle experiment: replace Langevin sampling with real MC negatives. Establishes upper bound on contrastive objective. |
| `latent_dim_variation/` | Sweep of AE models at latent dims 10, 20, 30, 40, 50, 60, 70, 80. Each subdir: `ae_zb_dim{N}/model_best.pkl`. |

Each checkpoint directory typically contains:
- `model_best.pkl` — PyTorch state dict (best by validation AUC or loss).
- `events.out.tfevents.*` — TensorBoard logs.

### `fast-ad/autoresearch/` — Autonomous Hyperparameter Search

An autonomous research agent that iteratively modifies `train.py` to maximize anomaly detection AUC. (Previously `nae-autoresearch/`.)

| File | Description |
|------|-------------|
| `train.py` | **The modifiable file.** Contains all NAE Phase 2 hyperparameters (GAMMA, NEG_LAMBDA, Z_STEPS, Z_STEP_SIZE, X_STEPS, X_NOISE_STD, etc.), Langevin sampling loops, replay buffer, loss function, and architecture. The agent modifies only this file. |
| `evaluate.py` | **Read-only evaluation harness.** Runs `train.py`, reads `metrics.json`, computes score = best_val_auc × stability_multiplier. Collapsed runs score 0.0. |
| `program.md` | Agent instructions: metric definition, failure modes, 12 hyperparameter categories to explore, research strategy, known instabilities. |
| `run_autoresearch.sh` | Slurm job script: submits evaluation harness to GPU, keeps node alive for interactive agent. |

### `fast-ad/plots/` — Generated Visualizations

| Subdirectory | Content |
|--------------|---------|
| `rocs_test/` | AE vs NAE ROC curves on held-out test split (per-signal + combined overlays). |
| `rocs_mc_oracle/` | Oracle NAE ROC curves; oracle-vs-teacher comparison plots. |
| `latent_dim_variation/` | Per-signal ROC curves across latent dims, mean-AUC-vs-dim summary, `auc_summary.csv`. |
| `training/` | Training curves (loss, AUC over epochs) from TensorBoard logs. |

---

## `slurm_scripts/` — HPC Job Submission

All scripts run on the adroit cluster (Slurm), load `anaconda3/2024.10` + `cudatoolkit/12.6`, and activate the conda env at `conda/envs/myenv`.

| Script | Description |
|--------|-------------|
| `run_training.sh` | Phase 1 AE + Phase 2 NAE training (24h walltime, 1 GPU, 64 GB RAM). |
| `run_classifiers.sh` | Runs both `train_et_regions_classifier.py` and `train_latent_classifier.py` sequentially (4h, GPU). |
| `run_lasso.sh` | Runs `lasso_analysis.py` (CPU-only, 4 cores, 16 GB RAM). |
| `eval_latent_dim_rocs.sh` | Evaluates ROC curves for all latent dim variations. |
| `run_ae_vs_nae_rocs.sh` | Generates AE vs NAE comparison ROC plots. |
| `run_nae_mc_oracle_rocs.sh` | Runs `nae_mc_oracle_rocs.py` on a GPU node. |
| `run_process.sh` | Runs `process_to_hdf5.py` (EOS data pipeline). |
| `skim_all.sh` | Re-skims all event types from CICADANtuples on CERN EOS. |
| `train_latent_dim_sweep.sh` | Trains AE at multiple latent dimensions (parameter sweep). |
| `slurm_test.sh` | Minimal test to verify environment. |

---

## `plots/` — Root-Level Generated Plots

| Subdirectory | Content |
|--------------|---------|
| `lasso/` | LASSO regularization paths, active sets, R^2 per observable. |
| `correlations/` | Observable correlation heatmaps across classes. |
| `et_regions_classifier/` | Confusion matrices, ROC curves, feature importances (et_regions classifiers). |
| `latent_classifier/` | Same for latent-space classifiers. |
| `tsne/` | t-SNE embeddings of latent space. |
| `training/` | Training curve figures. |

---

## `logs/` — Slurm Job Logs (gitignored)

Stdout/stderr from all submitted jobs. Naming: `train_*.out/err`, `classifiers_*.out/err`, `lasso_*.out/err`, `rocs_*.out/err`, etc.

