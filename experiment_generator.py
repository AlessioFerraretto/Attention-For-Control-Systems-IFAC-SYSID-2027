import itertools
import os

# Template configuration
TEMPLATE = """
# @package _global_

transformer_seed: 1
data_seed: 1

epochs: 100
learning_rate: 1e-3
eval_every: 101

training_batch_size: VAR_1
validation_batch_size: 64
test_batch_size: 64

save_model: True
save_checkpoints: True

model:
  description: a transformer architecture

  model_name: transformer # Name of the layer used
  embedding: transformer-positional-encoding # Embedding type
  layer_name: transformer-encoder-layer # Name of the layer used
  out_layer: token-wise-readout-for-impulse-response-prediction # Output layer type
  num_layers: VAR_2

  # Layer parameters
  input_dim: 2
  model_dim: VAR_3
  num_heads: VAR_4
  output_dim: ${model.model_dim}
  activation_function: gelu
  dropout: VAR_5

  # Only for out_layer = mlp-classifier
  input_dim_classifier: "${model.output_dim}" # Input dimension of the classifier = output_dim
  hidden_dim_classifier: "${model.output_dim}" # Hidden dimension of the classifier = output_dim
  output_dim_classifier: 1 # Output dimension of the classifier = 1
  out_bias1_classifier: True # Biases of classifier hidden layer and out layer
  out_bias2_classifier: True # Biases of classifier hidden layer and out layer

  # Only for out_layer = mean-pooled-classifier
  head_bias: True # Biases of the head layer
  eps: 1e-6 # Epsilon for numerical stability in the head layer

  out_dir_name: "transformer"

dataset:
  name: dynamic-impulse-response-prediction
  description: stable spline kernel generator of the type [ [u, y], padding] -> [g]

  shuffle: False # The sequences are already generated randomly

  batches_per_epoch: 1000

  g_length: 64
  n: 100
  n_past: 0
  n_future: "${dataset.n}"
  c_min: 0.0001
  c_max: 1.0
  alpha_min: 0.7
  alpha_max: 0.95
  snr_min: 3
  snr_max: 10
  jitter_regularization_parameter: 0.00001

  padding: 0
  filter_type: VAR_6
  a_param: 0.9

  out_dir_name: "dynamic-impulse-response-prediction"

wandb:
  enable: False
  project: AttentionForControlSystems
  entity: alessioferraretto-unipd
  dataset: dynamic-impulse-response-prediction
  model_name: "transformer"


"""


# Define the parameter grids
VAR_1 = [64, 256]
VAR_2 = [2, 4, 16]
VAR_3 = [64, 256]
VAR_4 = [4, 8, 32]
VAR_5 = [0, 0.1]
VAR_6 = ["white", "type_1", "type_2"]
# Create an output directory so we don't clutter the root folder
output_dir = "configs/experiment/bulk/"

os.makedirs(output_dir, exist_ok=True)

# Generate all possible combinations using itertools.product
combinations = list(itertools.product(VAR_1, VAR_2, VAR_3, VAR_4, VAR_5, VAR_6))

print(f"Generating {len(combinations)} configuration files...")

input("Press Enter to continue or Ctrl+C to abort...")

count = 0

# Iterate over each combination, replace variables, and save
for v1, v2, v3, v4, v5, v6 in combinations:
    # 1. Replace the placeholders in the template with the current values
    config_content = (
        TEMPLATE.replace("VAR_1", str(v1))
        .replace("VAR_2", str(v2))
        .replace("VAR_3", str(v3))
        .replace("VAR_4", str(v4))
        .replace("VAR_5", str(v5))
        .replace("VAR_6", str(v6))
    )
    count += 1
    # 2. Format the filename according to your required style
    filename = f"dynamic-impulse-response-prediction-snr3-10-{v1}-{v2}-{v3}-{v4}-{v5}-{v6}.yaml"
    filepath = os.path.join(output_dir, filename)

    # 3. Write the new configuration to the file
    with open(filepath, "w") as f:
        f.write(config_content)

print(f"Done! All files have been saved to the '{output_dir}' directory.")
