#!/bin/bash
#SBATCH --job-name=nae-oracle-rocs
#SBATCH --output=/scratch/network/lo8603/thesis/logs/nae_mc_oracle_rocs_%j.out
#SBATCH --error=/scratch/network/lo8603/thesis/logs/nae_mc_oracle_rocs_%j.err
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00

module purge
module load anaconda3/2024.10
module load cudatoolkit/12.6

conda activate /scratch/network/lo8603/thesis/conda/envs/myenv

echo "=============================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=============================="

nvidia-smi

cd /scratch/network/lo8603/thesis/fast-ad

python nae_mc_oracle_rocs.py

echo "=============================="
echo "End time: $(date)"
echo "=============================="
