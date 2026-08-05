#!/bin/bash

#SBATCH --ntasks 1
#SBATCH --partition allgroups
#SBATCH --time 24:00:00

#SBATCH --mem 24G
#SBATCH --cpus-per-task 4
#SBATCH --gres=gpu:rtx

#SBATCH --job-name AFCS_GridSearch
#SBATCH --array=0-575  # Spawns 576 jobs, numbered 0 through 575
#SBATCH --output outputs/dynamic-future-prediction-v3/transformer/%A_%a_out.txt
#SBATCH --error outputs/dynamic-future-prediction-v3/transformer/%A_%a_err.txt
#SBATCH --mail-type ALL

# 1. Define the folder containing all generated configs
CONFIG_DIR="configs/experiment/"

# 2. Read all .yaml files in the directory into a bash array
CONFIG_FILES=(${CONFIG_DIR}/*.yaml)

# 3. Select the specific file for THIS job using the array task ID
CURRENT_CONFIG=${CONFIG_FILES[$SLURM_ARRAY_TASK_ID]}

# 4. Extract just the filename without the .yaml extension
CONFIG_NAME=$(basename "$CURRENT_CONFIG" .yaml)

echo "Starting array task $SLURM_ARRAY_TASK_ID with config: $CONFIG_NAME"

# 5. Run the singularity container with the specific experiment config
srun singularity exec --nv singularity/afcs.sif python training_script.py experiment=$CONFIG_NAME
