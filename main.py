import os
import gc
import time
import pickle
import logging
import warnings

import numpy as np
import torch
import hydra
from omegaconf import DictConfig, open_dict
from torch.nn.parallel import DistributedDataParallel as DDP

from dataset.dataloader import init_dataloader
from utils.utils import init_setting, seed_setting
from training.training import build_model, Train
from utils.logger import Logger

warnings.filterwarnings('ignore')
logging.getLogger("torch.nn.parallel.distributed").setLevel(logging.WARNING)
logging.getLogger("torch.distributed.distributed_c10d").setLevel(logging.WARNING)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    init_setting(cfg)

    dataset = dict(np.load(cfg.dataset.data_path, allow_pickle=True))
    with open(cfg.dataset.repeated_idx_path, "rb") as f:
        fold_idx = pickle.load(f)

    logger = Logger(cfg)
    logger.init_logging() if cfg.local_rank == 0 else None

    iter_nums = list(cfg.iter_num) if cfg.iter_num else range(cfg.n_iter)
    for iteration in iter_nums:
        seed_setting(cfg)
        if cfg.local_rank == 0:
            print(f"<<<<<<<<<<<<< Iter [{iteration + 1}/{cfg.n_iter}] >>>>>>>>>>>>>")
            logger.init_iter_logging(iteration)

        iter_results = []
        fold_nums = list(cfg.fold_num) if cfg.fold_num else range(cfg.n_fold)
        for fold in fold_nums:
            seed_setting(cfg)
            if cfg.local_rank == 0:
                print(f"------- Fold [{fold + 1}/{cfg.n_fold}] Iter [{iteration + 1}/{cfg.n_iter}] -------")

            dataloaders = init_dataloader(cfg, iteration, fold, fold_idx, dataset)
            model = build_model(cfg)
            if cfg.distributed:
                model = DDP(model, device_ids=[cfg.local_rank], find_unused_parameters=False)

            train_model = Train(cfg, iteration, fold, model, dataloaders, logger)
            final_results = train_model.train()
            iter_results.append(final_results)

            gc.collect()
            torch.cuda.empty_cache()

        logger.log_iter(iteration, iter_results) if cfg.local_rank == 0 else None

        gc.collect()
        torch.cuda.empty_cache()

    logger.log_avg() if cfg.local_rank == 0 else None

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
