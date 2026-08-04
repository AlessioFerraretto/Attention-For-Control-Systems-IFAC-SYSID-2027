import itertools
import os

# Template configuration
TEMPLATE = """
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

srun singularity exec --nv singularity/afcs.sif python training_script.py experiment=dynamic-v3-VAR_1

"""


# Define the parameter grids
VAR_1 = [i for i in range(1, 32)]

# Create an output directory so we don't clutter the root folder
output_dir = "configs/experiment/slurms/"

os.makedirs(output_dir, exist_ok=True)

# Generate all possible combinations using itertools.product
combinations = list(itertools.product(VAR_1))

print(f"Generating {len(combinations)} configuration files...")

count = 0

# Iterate over each combination, replace variables, and save
for v1 in combinations:
    # 1. Replace the placeholders in the template with the current values
    config_content = TEMPLATE.replace("VAR_1", str(v1))
    count += 1
    # 2. Format the filename according to your required style
    filename = f"dynamic-v3-{count}.slurm"
    filepath = os.path.join(output_dir, filename)

    # 3. Write the new configuration to the file
    with open(filepath, "w") as f:
        f.write(config_content)

print(f"Done! All files have been saved to the '{output_dir}' directory.")
