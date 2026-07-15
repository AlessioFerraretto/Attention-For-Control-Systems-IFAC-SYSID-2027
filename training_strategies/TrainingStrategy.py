from training_strategies.interfaces import ModelComponent, TaskComponent


class TrainingStrategy:
    def __init__(self, task: TaskComponent, model_comp: ModelComponent):
        self.task = task
        self.model_comp = model_comp

    def loss_fn(self):
        return self.task.loss_fn()

    def compute_metrics(self, preds, labels):
        return self.task.compute_metrics(preds, labels)

    def extra_loss(self, model, cfg, global_step):
        return self.model_comp.extra_loss(
            model, cfg, global_step
        )  # TODO: Check if this is correct

    def setup(self, model, cfg, device, train_loader):
        return self.model_comp.setup(model, cfg, device, train_loader)

    def setup_for_eval(self, model):
        return self.model_comp.setup_for_eval(model)

    def make_optimizer(self, model, cfg):
        return self.model_comp.make_optimizer(model, cfg)

    def clip_gradients(self, model):
        return self.model_comp.clip_gradients(model)

    def before_step(self, model, global_step):
        return self.model_comp.before_step(model, global_step)

    def enter_eval(self, model):
        return self.model_comp.enter_eval(model)

    def exit_eval(self, model):
        return self.model_comp.exit_eval(model)

    def logging_metrics(self, model):  # , precomputed):
        return self.model_comp.logging_metrics(model)  # , precomputed)
