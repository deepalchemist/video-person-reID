from torch.optim.lr_scheduler import _LRScheduler


# update learning rate (called once every epoch)
def update_lr(scheduler, epoch, n_iter=None):
    if n_iter is None:
        scheduler.step(epoch)
    else:
        scheduler.step_iter_wise(epoch, n_iter)  # iter-wise warm-up


class WarmUpLR(_LRScheduler):
    def __init__(self,
                 optimizer,
                 scheduler,
                 warmup_epochs,
                 total_batches,
                 mode="linear",
                 alpha=0.01,
                 last_epoch=-1):
        self.mode = mode
        self.scheduler = scheduler
        self.warmup_epochs = warmup_epochs
        self.alpha = alpha
        self.n_batch_per_epoch = total_batches

        super(WarmUpLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            if self.mode == "linear":
                beta = self.last_epoch / float(self.warmup_epochs)  # 0 -> 1
                factor = self.alpha * (1 - beta) + beta  # alpha -> 1
            elif self.mode == "constant":
                factor = self.alpha
            else:
                raise KeyError("WarmUp type {} not implemented".format(self.mode))

            return [factor * base_lr for base_lr in self.base_lrs]  # initial lr
            # return [factor * base_lr for base_lr in cold_lrs]
        else:
            self.scheduler.last_epoch = self.last_epoch - self.warmup_epochs  # NOTE: important
            cold_lrs = self.scheduler.get_lr()

            return cold_lrs

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def get_lr_iter_wise(self, n_iter):
        if self.last_epoch < self.warmup_epochs:
            if self.mode == "linear":
                beta = n_iter / float(self.warmup_epochs * self.n_batch_per_epoch)  # 0 -> 1
                factor = self.alpha * (1 - beta) + beta  # alpha -> 1
            elif self.mode == "constant":
                factor = self.alpha
            else:
                raise KeyError("WarmUp type {} not implemented".format(self.mode))

            return [factor * base_lr for base_lr in self.base_lrs]  # initial lr
            # return [factor * base_lr for base_lr in cold_lrs]
        else:
            self.scheduler.last_epoch = self.last_epoch - self.warmup_epochs  # NOTE: important
            cold_lrs = self.scheduler.get_lr()

            return cold_lrs

    def step_iter_wise(self, epoch=None, n_iter=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr_iter_wise(n_iter)):
            param_group['lr'] = lr
