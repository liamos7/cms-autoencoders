#!/bin/bash
#SBATCH --job-name=gpu-test
#SBATCH --output=/scratch/network/lo8603/thesis/logs/gpu-test_%j.out
#SBATCH --error=/scratch/network/lo8603/thesis/logs/gpu-test_%j.err
#SBATCH --nodelist=adroit-h11g1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00

module purge
module load cudatoolkit/12.6

conda activate /scratch/network/lo8603/thesis/conda/envs/myenv

nvidia-smi
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

python /scratch/network/lo8603/thesis/test_imports.py