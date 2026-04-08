#!/bin/bash
#SBATCH --job-name=train-teacher
#SBATCH --output=/scratch/network/lo8603/thesis/logs/train_%j.out
#SBATCH --error=/scratch/network/lo8603/thesis/logs/train_%j.err
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# Load modules
module purge
module load anaconda3/2024.10
module load cudatoolkit/12.6

# Activate environment
conda activate /scratch/network/lo8603/thesis/conda/envs/myenv

# Print job info
echo "=============================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=============================="

# Print GPU info
nvidia-smi

# Print Python/CUDA info
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Navigate to the directory containing train-teacher.py
cd /scratch/network/lo8603/thesis/fast-ad

# Run training
python train-teacher.py \
    -ds "CICADA" \
    --data-root-path "./data/h5_files/" \
    --model "NAEWithEnergyTraining" \
    --load-pretrained-path "./outputs/latent_dim_variation/ae_zb_dim20/model_best.pkl" \
    -o "./outputs/nae_phase2_tuned_dim20" \
    -ho 1,2,3,4,5,6,7,8,9,10 \
    --epochs 100 \
    -v

echo "=============================="
echo "End time: $(date)"
echo "=============================="
