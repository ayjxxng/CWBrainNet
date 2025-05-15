import os
import math
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig


class LRScheduler:
    def __init__(self, cfg: DictConfig, optimizer_cfg: DictConfig) -> None:
        self.lr_cfg = optimizer_cfg.lr_scheduler
        self.train_cfg = cfg
        self.lr = optimizer_cfg.lr
        assert self.lr_cfg.mode in ['cos', 'linear'], "LRScheduler: invalid mode."

    def update(self, optimizer: torch.optim.Optimizer, step: int) -> None:
        base_lr = self.lr_cfg.base_lr
        target_lr = self.lr_cfg.base_lr * self.lr_cfg.target_rate

        warm_up_from = self.lr_cfg.warm_up_from
        warm_up_steps = self.lr_cfg.warm_up_steps
        total_steps = self.train_cfg.total_steps

        assert 0 <= step <= total_steps, f"step:{step}, total_step:{total_steps}"

        if step < warm_up_steps:
            ratio = step / warm_up_steps
            self.lr = warm_up_from + (base_lr - warm_up_from) * ratio
        else:
            ratio = (step - warm_up_steps) / (total_steps - warm_up_steps)
            if self.lr_cfg.mode == 'cos':
                cosine = math.cos(math.pi * ratio)
                self.lr = target_lr + (base_lr - target_lr) * (1 + cosine) / 2
            elif self.lr_cfg.mode == 'linear':
                self.lr = target_lr + (base_lr - target_lr) * (1 - ratio)

        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr


class Checkpoint:
    def __init__(
            self, patience: int = 5, delta: float = 1e-5, distributed: bool = False
    ) -> None:
        self.patience = patience
        self.delta = delta
        self.distributed = distributed

        self.best_score = None
        self.score_max = -np.inf
        self.counter = 0
        self.early_stop = False

    def __call__(
            self, score: float, epoch: int, optimizer: torch.optim.Optimizer,
            lr_schedulers: LRScheduler, model: torch.nn.Module, model_path: str, rank: int
    ) -> None:
        self.model_path = model_path
        self.epoch = epoch
        self.optimizer = optimizer
        self.lr = lr_schedulers.lr

        if self.best_score is None:
            self.best_score = score
            if rank == 0:
                self.save_checkpoint(score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if rank == 0:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            if rank == 0:
                self.save_checkpoint(score, model)
        if self.distributed:
            dist.barrier()

    def save_checkpoint(self, score: float, model: torch.nn.Module) -> None:
        print(f'Validation AUC increased ({self.score_max:.5f} --> {score:.5f}).  Saving model ...')
        torch.save({
            'epoch': self.epoch,
            'model': model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr': self.lr,
        }, self.model_path)
        self.score_max = score
      
