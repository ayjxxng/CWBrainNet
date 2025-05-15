import os
import warnings
from typing import List, Tuple, Optional, Dict, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from omegaconf import DictConfig, open_dict


class CWT_Dataset(Dataset):
    def __init__(
            self, dataset: np.ndarray,
            labels: np.ndarray, sub_id: Optional[np.ndarray] = None,
            site: Optional[np.ndarray] = None
    ) -> None:
        self.dataset = torch.stack(
            [torch.FloatTensor(dataset.real), torch.FloatTensor(dataset.imag)]
            , dim=-1
        )
        self.labels = F.one_hot(torch.LongTensor(labels))
        self.len = self.labels.shape[0]
        self.sub_id = sub_id
        self.site = site

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        data = {
            'inputs': self.dataset[index],
            'labels': self.labels[index]
        }
        if self.sub_id is not None:
            data['sub_id'] = self.sub_id[index]
        if self.site is not None:
            data['site'] = self.site[index]
        return data

    def __len__(self) -> int:
        return self.len


def get_dataloader(
        cfg: DictConfig, samples: np.ndarray,
        labels: np.ndarray = None, sub_id: Optional[np.ndarray] = None,
        site: Optional[np.ndarray] = None, shuffle: bool = True
) -> DataLoader:
    dataset = CWT_Dataset(
        dataset=samples,
        labels=labels,
        sub_id=sub_id,
        site=site
    )
    sampler = DistributedSampler(
        dataset=dataset,
        shuffle=shuffle
    ) if cfg.distributed else None
    return DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False
    )


def init_dataloader(
        cfg: DictConfig, iteration: int, fold: int,
        fold_idx: Dict, dataset: Dict[str, np.ndarray]
) -> List[DataLoader]:
    key = cfg.dataset.data_key
    data_idx = fold_idx[f'iter{iteration + 1}'][f'fold{fold + 1}']

    train_loader = get_dataloader(
        cfg=cfg,
        samples=dataset[key][data_idx['train']],
        labels=dataset['label'][data_idx['train']],
        shuffle=True
    )
    valid_loader = get_dataloader(
        cfg=cfg,
        samples=dataset[key][data_idx['valid']],
        labels=dataset['label'][data_idx['valid']],
        shuffle=False
    )
    test_loader = get_dataloader(
        cfg=cfg,
        samples=dataset[key][data_idx['test']],
        labels=dataset['label'][data_idx['test']],
        shuffle=False
    )
    with open_dict(cfg):
        cfg.steps_per_epoch = len(train_loader)
        cfg.total_steps = cfg.steps_per_epoch * cfg.training.train_epochs
    return [train_loader, valid_loader, test_loader]


def continuous_mixup_data(
        *xs: torch.Tensor, y: Optional[torch.Tensor] = None,
        alpha: float = 1.0, device: str = 'cuda'
) -> Tuple[torch.Tensor, ...]:
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = xs[0].size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_xs = [lam * x + (1 - lam) * x[index, :] for x in xs]
    if y is not None:
        y = lam * y + (1 - lam) * y[index]
    return *mixed_xs, y
