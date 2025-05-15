import os
import warnings
from typing import List, Dict

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryConfusionMatrix
from torchmetrics.aggregation import SumMetric
from tqdm.auto import tqdm
from omegaconf import DictConfig, open_dict

from models.CWBrainNet import CWBrainNet
from dataset.dataloader import continuous_mixup_data
from utils.lr_scheduler import Checkpoint, LRScheduler
from utils.logger import Logger

warnings.filterwarnings('ignore')


class Train:
    def __init__(
            self, cfg: DictConfig, iteration: int, fold: int,
            model: nn.Module, dataloaders: List[DataLoader], logger: Logger
    ) -> None:
        self.cfg = cfg
        self.iter = iteration
        self.fold = fold
        self.model = model
        self.dataloaders = dataloaders  # [train, valid, test]
        self.logger = logger
        self.n_gpu = cfg.n_gpu
        self.rank = cfg.local_rank
        self.distributed = cfg.distributed
        self.current_step = 0

        self.iter_path = os.path.join(cfg.path, f'Iter_{self.iter + 1}')
        self.model_path = os.path.join(self.iter_path, f'iter{self.iter + 1}_fold{self.fold + 1}_model.pt')

        if self.rank == 0:
            os.makedirs(self.iter_path, exist_ok=True)
            self.logger.init_fold_logging(iteration, fold)

    def train_per_epoch(self, dataloader, optimizer, lr_scheduler) -> Dict[str, float]:
        self.model.train()
        loss_metric = SumMetric().cuda(self.rank)
        acc_metric = BinaryAccuracy().cuda(self.rank)

        for batch in tqdm(dataloader, leave=True, disable=self.rank != 0):
            self.current_step += 1
            lr_scheduler.update(optimizer, step=self.current_step)

            for key, value in batch.items():
                batch[key] = value.cuda(self.rank)

            if self.cfg.training.mixup_data:
                batch['inputs'], batch['labels'] = continuous_mixup_data(batch['inputs'], y=batch['labels'])

            results = self.model(batch)
            optimizer.zero_grad()
            results['loss'].backward()
            optimizer.step()

            loss_metric.update(results['loss'].item())
            acc_metric.update(results['probs'], batch['labels'].argmax(dim=-1))

        train_results = {
            'loss': loss_metric.compute().item() / (len(dataloader) * self.n_gpu),
            'acc': acc_metric.compute().item()
        }
        loss_metric.reset()
        acc_metric.reset()
        return train_results

    def evaluate(self, dataloader) -> Dict[str, float]:
        self.model.eval()
        loss_metric = SumMetric().cuda(self.rank)
        metrics = MetricCollection({
            'auc': BinaryAUROC(),
            'acc': BinaryAccuracy(),
            'mat': BinaryConfusionMatrix(),
        }).cuda(self.rank)

        with torch.no_grad():
            for batch in dataloader:
                for key, value in batch.items():
                    batch[key] = value.cuda(self.rank)

                results = self.model(batch)
                loss_metric.update(results['loss'].item())
                metrics.update(results['probs'], batch['labels'].argmax(dim=-1))

        eval_result = metrics.compute()
        TN, FP, FN, TP = eval_result['mat'].flatten().tolist()
        eval_results = {
            'loss': loss_metric.compute().item() / (len(dataloader) * self.n_gpu),
            'auc': eval_result['auc'].item(),
            'acc': eval_result['acc'].item(),
            'sen': TP / (TP + FN) if (TP + FN) > 0 else 0,
            'spc': TN / (TN + FP) if (TN + FP) > 0 else 0,
        }
        metrics.reset()
        loss_metric.reset()
        return eval_results

    def test(self, dataloader) -> Dict[str, float]:
        if self.rank == 0:
            print('Loading best model...')
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank} if self.distributed else None
        checkpoint = torch.load(self.model_path, map_location=map_location)
        self.model.load_state_dict(checkpoint['model'])

        results = self.evaluate(dataloader)
        if self.rank == 0:
            self.logger.log_fold(self.iter, self.fold, results)
        return results

    def train(self) -> Dict[str, float]:
        optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.optimizer.lr,
                               weight_decay=self.cfg.optimizer.weight_decay)
        lr_schedulers = LRScheduler(cfg=self.cfg, optimizer_cfg=self.cfg.optimizer)
        check_epoch = Checkpoint(patience=self.cfg.training.patience, distributed=self.distributed)

        for epoch in range(self.cfg.training.train_epochs):
            if self.rank == 0:
                print(f"Epoch[{epoch + 1}/{self.cfg.training.train_epochs}] ========================")

            if self.distributed:
                for loader in self.dataloaders:
                    loader.sampler.set_epoch(epoch)

            train_results = self.train_per_epoch(self.dataloaders[0], optimizer, lr_schedulers)
            valid_results = self.evaluate(self.dataloaders[1])
            test_results = self.evaluate(self.dataloaders[2])

            if self.rank == 0:
                self.logger.log_epoch(epoch, train_results, valid_results, test_results)

            check_epoch(valid_results['auc'], epoch, optimizer, lr_schedulers, self.model, self.model_path, self.rank)
            if check_epoch.early_stop:
                if self.rank == 0:
                    print("Early stopping")
                break

        if self.distributed:
            dist.barrier()

        final_results = self.test(self.dataloaders[2])
        return final_results


def build_model(cfg: DictConfig) -> nn.Module:
    model = CWBrainNet(cfg).cuda(cfg.local_rank)

    if cfg.local_rank == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Trainable Params: {total_params:,}")

        with open_dict(cfg):
            cfg.total_params = total_params

    return model
