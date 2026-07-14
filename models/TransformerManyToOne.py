import torch
import torch.nn as nn


class SystemIdTransformer(nn.Module):
    def __init__(
        self, seq_len, g_len, future_horizon, d_model=64, nhead=4, num_layers=3
    ):
        super().__init__()

        # Input features: u_t and y_t at each timestep (dim=2)
        self.input_proj = nn.Linear(2, d_model)

        # Positional encoding (learned for simplicity here)
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, d_model))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Task 1: Impulse Response Estimation Head (g_1:m)
        self.g_head = nn.Sequential(
            nn.Linear(d_model * seq_len, 128), nn.ReLU(), nn.Linear(128, g_len)
        )

        # Task 2: Future Output Prediction Head (y_N+1:N+H)
        # Note: We pass future inputs 'u_future' into this head as well to condition the prediction
        self.y_head = nn.Sequential(
            nn.Linear((d_model * seq_len) + future_horizon, 128),
            nn.ReLU(),
            nn.Linear(128, future_horizon),
        )

    def forward(self, u_past, y_past, u_future):
        # Combine inputs into tokens: shape (batch_size, seq_len, 2)
        inputs = torch.stack([u_past, y_past], dim=-1)

        # Project and add positional encoding
        x = self.input_proj(inputs) + self.pos_encoder

        # Pass through Transformer
        encoded = self.transformer(x)

        # Flatten for multi-task heads
        encoded_flat = encoded.flatten(start_dim=1)

        # Task 1 prediction
        g_hat = self.g_head(encoded_flat)

        # Task 2 prediction (conditioned on future inputs)
        y_context = torch.cat([encoded_flat, u_future], dim=1)
        y_hat_future = self.y_head(y_context)

        return g_hat, y_hat_future
