#!/bin/bash
#SBATCH --job-name=lasso_analysis
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/lasso_%j.out
#SBATCH --error=logs/lasso_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lo8603@princeton.edu

mkdir -p logs

cd /scratch/network/lo8603/thesis

/scratch/network/lo8603/thesis/conda/envs/myenv/bin/python lasso_analysis.py \
    --n_events 500000 \
    --n_events_r2 200000 \
    --out_dir plots/lasso
