#!/bin/bash --login
#
#SBATCH --job-name=uci_regression_fs
#SBATCH --time=1-00:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --array=1-9

# Read the line corresponding to the current task ID
values=$(grep "^${SLURM_ARRAY_TASK_ID}:" regression_datasets.txt)
dataset=$(echo $values | cut -f 2 -d:)

# Activate the environment
module purge
conda activate myenv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
srun python uci_regression_fs.py --name=$dataset