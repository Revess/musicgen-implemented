#!/bin/bash
#SBATCH --job-name=musiclm
#SBATCH --time=12:00:00
#SBATCH -N 1
#SBATCH -p gpu_h100
#SBATCH --gpus 1
#SBATCH --output=./jobs/%x_%A_%a.out    # Standard output and error log

. /etc/bashrc
. ~/.bashrc
module load 2023
module load CUDA/12.1.1
conda activate musiclm

echo $$

python ./trainings.py train_mulan