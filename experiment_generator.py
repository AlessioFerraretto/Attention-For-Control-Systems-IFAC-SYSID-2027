import itertools
import os

# Template configuration
TEMPLATE = """
# @package _global_

running_mode: inference
model_mode: load
model_path: "/home/ferraretto/Desktop/Attention For Control Systems/saved_models/model_VAR_1_VAR_2_VAR_3.pt"

transformer_seed: 1
data_seed: 1

epochs: 50
learning_rate: 1e-3
eval_every: 50

training_batch_size: 64
validation_batch_size: 64
test_batch_size: 64

save_model: True
save_checkpoints: False

model:
  description: a transformer architecture

  model_name: transformer # Name of the layer used
  embedding: transformer-positional-encoding # Embedding type
  layer_name: transformer-encoder-layer # Name of the layer used
  out_layer: token-wise-readout-for-impulse-response-prediction # Output layer type
  num_layers: VAR_1

  # Layer parameters
  input_dim: 2
  model_dim: VAR_2
  num_heads: VAR_3
  output_dim: ${model.model_dim}
  activation_function: gelu
  dropout: 0.1

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

  kernel: "stable-spline"

  shuffle: False # The sequences are already generated randomly

  batches_per_epoch: 1000

  g_length: 64
  n: 100
  n_past: 0
  n_future: "${dataset.n}"
  c_min: 0.0001
  c_max: 1.0
  alpha_min: 0.70
  alpha_max: 0.95
  snr_min: 3
  snr_max: 10
  jitter_regularization_parameter: 0.00001

  padding: 0
  filter_type: "type_1"
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
VAR_1 = [4, 8]
VAR_2 = [64, 128]
VAR_3 = [16, 32]
# Create an output directory so we don't clutter the root folder
output_dir = "configs/experiment/bulk/"

os.makedirs(output_dir, exist_ok=True)

# Generate all possible combinations using itertools.product
combinations = list(itertools.product(VAR_1, VAR_2, VAR_3))

print(f"Generating {len(combinations)} configuration files...")

input("Press Enter to continue or Ctrl+C to abort...")

count = 0

# Iterate over each combination, replace variables, and save
for v1, v2, v3 in combinations:
    # 1. Replace the placeholders in the template with the current values
    config_content = (
        TEMPLATE.replace("VAR_1", str(v1))
        .replace("VAR_2", str(v2))
        .replace("VAR_3", str(v3))
    )
    count += 1
    # 2. Format the filename according to your required style
    filename = f"dynamic-impulse-response-prediction-snr3-10-{v1}-{v2}-{v3}.yaml"
    filepath = os.path.join(output_dir, filename)

    # 3. Write the new configuration to the file
    with open(filepath, "w") as f:
        f.write(config_content)

print(f"Done! All files have been saved to the '{output_dir}' directory.")
