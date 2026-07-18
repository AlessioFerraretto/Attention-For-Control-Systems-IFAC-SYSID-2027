from torch.optim import AdamW, Optimizer

from utils.Metrics import collect_metrics


class Transformer:
    def __init__(self):
        pass

    def setup(self, model, cfg, device, train_loader):
        pass

    def setup_for_eval(self, model):
        pass

    def extra_loss(self, model, cfg, global_step):
        return 0  # TODO: Check if this is correct

    def make_optimizer(self, model, cfg) -> Optimizer:
        return AdamW(model.parameters(), lr=cfg.learning_rate)

    def clip_gradients(self, model):
        return None

    def before_step(self, model, global_step):
        return None

    def enter_eval(self, model):
        return None

    def exit_eval(self, model):
        return None

    def logging_metrics(self, model):
        """Extract all metrics and produce and create a dictionary that is going
        to be sent to wandb and a string that will be used to log on the screen. The idea of this
        method is that, among all metrics , we filter out the one we want to print on the screen
        and the one we want to log in wandb"""

        # Collect all available metrics coming from the model
        log_metrics = collect_metrics(model)

        # TODO: Check if this is correct

        return log_metrics, ""
