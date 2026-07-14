from torch.utils.data import Dataset

from utils.StableSplineGenerator import generate_stable_spline


class StableSplineDataset(Dataset):
    def __init__(
        self,
        batches_per_epoch,
        batch_size,
        sequence_length,
        impulse_length,
        future_horizon,
        c_min,
        c_max,
        alpha_min,
        alpha_max,
        snr_min,
        snr_max,
    ):
        """
        Generates synthetic data for system identification based on Stable Spline Priors.
        """

        self.data = generate_stable_spline(
            impulse_length,
            # ...
            c_min,
            c_max,
            alpha_min,
            alpha_max,
            snr_min,
            snr_max,
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]
