#!/bin/bash

#SBATCH --ntasks 1
#SBATCH --partition allgroups
#SBATCH --time 24:00:00

#SBATCH --mem 24G
#SBATCH --cpus-per-task 4
#SBATCH --gres=gpu:rtx

#SBATCH --job-name AttentionForControlSystems
#SBATCH --output outputs/dynamic-future-prediction-v3/transformer/%j/slurm-out.txt
#SBATCH --error outputs/dynamic-future-prediction-v3/transformer/%j/slurm-err.txt
#SBATCH --mail-type ALL

srun singularity exec --nv singularity/afcs.sif python training_script.py experiment=dynamic-v3-1
