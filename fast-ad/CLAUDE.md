# Thesis: Anomaly Detection for CICADA L1 Trigger

## Project
Senior thesis at Princeton. Anomaly detection for CMS/LHC CICADA trigger system.
Goal: outperform CICADA autoencoder at detecting BSM physics (SUEP, GGH2TT, etc) against zero-bias background.

## Environment
- Compute: adroit-h11g1 GPU node (NOT the login node where claude runs)
- Conda env: /scratch/network/lo8603/thesis/conda/envs/myenv
- Data: data/h5_files/
- Key scripts: nae-autoresearch/train.py, nae-autoresearch/evaluate.py
- Agent instructions: nae-autoresearch/program.md
- To submit jobs: bash nae-autoresearch/run_autoresearch.sh

## Structure
- fastad/: core modules (datasets.py, modules.py, teachers.py, etc)
- nae-autoresearch/: NAE training + evaluation + autoresearch loop
- outputs/: results
- rocs.py, train-teacher.py: analysis scripts

## Current focus
NAE Phase 2 energy-based training with Langevin Monte Carlo in 80-dim spherical latent space.
Hyperparameter tuning ongoing. See nae-autoresearch/program.md for full context and known failure modes.

## Constraints
- Claude Code runs on LOGIN node (adroit5), not GPU node
- Never run training directly — always submit via run_autoresearch.sh or sbatch
