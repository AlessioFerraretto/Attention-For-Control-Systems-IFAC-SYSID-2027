import torch.nn as nn


class Transformer(nn.Module):
    def __init__(
        self,
        input_dim,
        model_dim,
        num_heads,
        activation_function,
        num_layers,
        output_dim,
    ):
        super().__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            batch_first=True,  # This ensures that the input and output tensors are of shape (batch_size, seq_len, model_dim)
            activation=activation_function,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )
        self.fc_out = nn.Linear(model_dim, output_dim)

    def forward(self, src):
        embedded = self.embedding(src)  # (seq_len, batch_size, model_dim)
        transformer_output = self.transformer_encoder(
            embedded
        )  # (seq_len, batch_size, model_dim)
        output = self.fc_out(transformer_output)  # (seq_len, batch_size, output_dim)
        return output
