import numpy as np
import torch
from torch.utils.data import Dataset

from utils.StableSplineGenerator import generate_stable_spline


class InductionHead(Dataset):
    def __init__(self, num_samples, seq_len, m):
        """Generates a static dataset of synthetic linear systems."""
        self.inputs = []
        self.targets = []

        for _ in range(num_samples):
            sample = generate_stable_spline(m=m, N=seq_len)
            u = sample["u"]
            y = sample["y"]
            g = sample["g"]

            # Stack u and y as features for the sequence: shape (seq_len, 2)
            src = np.stack([u, y], axis=1)

            self.inputs.append(torch.tensor(src, dtype=torch.float32))
            self.targets.append(torch.tensor(g, dtype=torch.float32))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
