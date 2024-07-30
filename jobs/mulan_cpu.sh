#!/bin/bash
#SBATCH --job-name=musiclm
#SBATCH --time=24:00:00
#SBATCH -C cpunode
#SBATCH --output=./jobs/%x_%A_%a.out    # Standard output and error log

. /etc/bashrc
. ~/.bashrc
module load cuda12.1/toolkit
conda activate musiclm

echo $$

python ./trainings.py train_mulan --device='cpu'