#!/bin/bash
#SBATCH --job-name=ae-vs-nae-rocs
#SBATCH --output=/scratch/network/lo8603/thesis/logs/rocs_%j.out
#SBATCH --error=/scratch/network/lo8603/thesis/logs/rocs_%j.err
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
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

python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

cd /scratch/network/lo8603/thesis/fast-ad

python ae_vs_nae_rocs.py

echo "=============================="
echo "End time: $(date)"
echo "=============================="
