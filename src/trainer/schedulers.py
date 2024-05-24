import math
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
import torch.optim.lr_scheduler as lr_scheduler
from pytorch_lightning.callbacks import LearningRateMonitor


class ScheduledOptimLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, d_model, n_warmup_steps, last_epoch=-1):
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.init_lr = np.power(d_model, -0.5)
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        lr = self.init_lr * self._get_lr_scale()
        return [lr for _ in self.optimizer.param_groups]

    def _get_lr_scale(self):
        return min(
            np.power(self.last_epoch, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.last_epoch
        )

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
