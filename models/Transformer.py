import torch
import torch.nn as nn

from monitor.TrackableModule import TrackableModule


class Transformer(TrackableModule):
    """A compact transformer block for sequence-to-sequence prediction."""

    def __init__(self, cfg, device, input_dim: int):
        super().__init__()

        self.device = device
        self.input_dim = input_dim
        self.model_dim = cfg.model.model_dim
        self.num_heads = cfg.model.num_heads
        self.num_layers = cfg.model.num_layers
        self.output_dim = cfg.model.output_dim
        self.activation_function = cfg.model.activation_function
        self.dropout = cfg.model.dropout

        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.model_dim % self.num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")

        self.input_proj = nn.Linear(
            self.input_dim, self.model_dim, device=device, dtype=torch.float
        )

        self.positional_encoding = nn.Parameter(
            torch.zeros(1, 1, self.model_dim, device=self.device)
        )

        self.blocks = nn.ModuleList(
            [self._build_block() for _ in range(self.num_layers)]
        )
        self.norm = nn.LayerNorm(self.model_dim, device=self.device, dtype=torch.float)
        self.readout = nn.Sequential(
            nn.Linear(
                self.model_dim,
                max(16, self.model_dim // 2),
                device=self.device,
                dtype=torch.float,
            ),
            self._activation(self.activation_function),
            nn.Linear(
                max(16, self.model_dim // 2),
                self.output_dim,
                device=self.device,
                dtype=torch.float,
            ),
        )

    def _build_block(self):
        return nn.ModuleDict(
            {
                "attn": nn.MultiheadAttention(
                    embed_dim=self.model_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                    batch_first=True,
                    device=self.device,
                    dtype=torch.float,
                ),
                "norm1": nn.LayerNorm(
                    self.model_dim,
                    device=self.device,
                    dtype=torch.float,
                ),
                "ffn": nn.Sequential(
                    nn.Linear(
                        self.model_dim,
                        4 * self.model_dim,
                        device=self.device,
                        dtype=torch.float,
                    ),
                    self._activation(self.activation_function),
                    nn.Linear(
                        4 * self.model_dim,
                        self.model_dim,
                        device=self.device,
                        dtype=torch.float,
                    ),
                ),
                "norm2": nn.LayerNorm(
                    self.model_dim,
                    device=self.device,
                    dtype=torch.float,
                ),
            }
        )

    def _activation(self, name):
        if name.lower() == "gelu":
            return nn.GELU()
        elif name.lower() == "tanh":
            return nn.Tanh()
        elif name.lower() == "relu":
            return nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation function: {name}")

    def _positional_embeddings(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device, dtype=x.dtype)
        div_term = torch.exp(
            torch.arange(0, self.model_dim, 2, device=x.device, dtype=x.dtype)
            * (-torch.log(torch.tensor(10000.0, device=x.device)) / self.model_dim)
        )
        pe = torch.zeros(seq_len, self.model_dim, device=x.device, dtype=x.dtype)
        pe[:, 0::2] = torch.sin(positions.unsqueeze(1) * div_term.unsqueeze(0))
        if self.model_dim > 1:
            pe[:, 1::2] = torch.cos(
                positions.unsqueeze(1) * div_term.unsqueeze(0)[: pe[:, 1::2].shape[1]]
            )
        return pe.unsqueeze(0).to(x.dtype)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(
                "Expected input of shape (batch, seq_len, features); "
                f"got {tuple(x.shape)}"
            )
        projected = self.input_proj(x)
        positional_encoding = self.positional_encoding.to(device=projected.device)
        positions = self._positional_embeddings(projected) + positional_encoding
        hidden = projected + positions

        for block in self.blocks:
            attn_out, _ = block["attn"](hidden, hidden, hidden, need_weights=False)
            hidden = block["norm1"](hidden + attn_out)
            feed_forward = block["ffn"](hidden)
            hidden = block["norm2"](hidden + feed_forward)

        hidden = self.norm(hidden)
        return self.readout(hidden)
