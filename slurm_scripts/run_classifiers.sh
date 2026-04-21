#!/bin/bash
#SBATCH --job-name=classifiers
#SBATCH --output=/scratch/network/lo8603/thesis/logs/classifiers_%j.out
#SBATCH --error=/scratch/network/lo8603/thesis/logs/classifiers_%j.err
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00

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

cd /scratch/network/lo8603/thesis

mkdir -p logs plots/et_regions_classifier plots/latent_classifier

echo ""
echo "=== Running train_et_regions_classifier.py ==="
python train_et_regions_classifier.py

echo ""
echo "=== Running train_latent_classifier.py ==="
python train_latent_classifier.py

echo ""
echo "=============================="
echo "End time: $(date)"
echo "=============================="
