import itertools
import os

# Template configuration
TEMPLATE = """

# @package _global_

seed: null

epochs: 100
learning_rate: 1e-3
eval_every: 1

training_batch_size: 256
validation_batch_size: 256
test_batch_size: 256

model:
  description: a transformer architecture

  model_name: transformer # Name of the layer used
  embedding: transformer-positional-encoding # Embedding type
  layer_name: transformer-encoder-layer # Name of the layer used
  out_layer: mlp-classifier-for-future-prediction-v3 # Output layer type
  num_layers: VAR_1

  # Layer parameters
  input_dim: 2
  model_dim: VAR_2
  num_heads: VAR_3
  output_dim: ${model.model_dim}
  activation_function: gelu
  dropout: 0.1

  # Only for out_layer = mlp-classifier
  input_dim_classifier: "${mul: ${dataset.n_future}, ${model.output_dim} }" # Input dimension of the classifier = n_future * output_dim
  hidden_dim_classifier: "${div: ${model.input_dim_classifier}, 2 }" # Hidden dimension of the classifier = input_dim_classifier/2
  output_dim_classifier: ${dataset.n_future} # Output dimension of the classifier = n_future
  out_bias1_classifier: True # Biases of classifier hidden layer and out layer
  out_bias2_classifier: True # Biases of classifier hidden layer and out layer

  # Only for out_layer = mean-pooled-classifier
  head_bias: True # Biases of the head layer
  eps: 1e-6 # Epsilon for numerical stability in the head layer

  out_dir_name: "transformer"

dataset:
  name: dynamic-future-prediction-v3
  description: stable spline kernel generator of the type [ [u_past, y_past], [u_future, padding] ] -> [y_future]

  shuffle: False # The sequences are already generated randomly

  batches_per_epoch: 1000

  g_length: VAR_4
  n_past: 90
  n_future: 10
  c_min: 0.0001
  c_max: 1.0
  alpha_min: 0.7
  alpha_max: 0.95
  snr_min: 100
  snr_max: 100
  jitter_regularization_parameter: 0.00001

  padding: 0

  out_dir_name: "dynamic-future-prediction-v3"

wandb:
  enable: False
  project: AttentionForControlSystems
  entity: alessioferraretto-unipd
  dataset: dynamic-future-prediction-v3
  model_name: "transformer"
"""


# Define the parameter grids
VAR_1 = [4, 8, 32, 64]
VAR_2 = [128, 256]
VAR_3 = [8, 16, 32, 64]
VAR_4 = [64]

# Create an output directory so we don't clutter the root folder
output_dir = "configs/experiment/bulk/"

os.makedirs(output_dir, exist_ok=True)

# Generate all possible combinations using itertools.product
combinations = list(itertools.product(VAR_1, VAR_2, VAR_3, VAR_4))

print(f"Generating {len(combinations)} configuration files...")

count = 0

# Iterate over each combination, replace variables, and save
for v1, v2, v3, v4 in combinations:
    # 1. Replace the placeholders in the template with the current values
    config_content = (
        TEMPLATE.replace("VAR_1", str(v1))
        .replace("VAR_2", str(v2))
        .replace("VAR_3", str(v3))
        .replace("VAR_4", str(v4))
    )
    count += 1
    # 2. Format the filename according to your required style
    filename = f"dynamic-future-prediction-v3-{v1}-{v2}-{v3}-{v4}.yaml"
    filepath = os.path.join(output_dir, filename)

    # 3. Write the new configuration to the file
    with open(filepath, "w") as f:
        f.write(config_content)

print(f"Done! All files have been saved to the '{output_dir}' directory.")
