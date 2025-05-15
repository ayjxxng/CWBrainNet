import os
import numpy as np
import torch
import torch.distributed as dist
from datetime import datetime
from omegaconf import DictConfig, open_dict


def init_setting(cfg: DictConfig) -> None:
    """Initialize the environment settings."""
    os.environ["HYDRA_FULL_ERROR"] = "1"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["PYTHONHASHSEED"] = str(cfg.seed)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(cfg.gpu))

    seed_setting(cfg)
    with open_dict(cfg):
        cfg.unique_id = datetime.now().strftime("%m-%d-%H-%M")
        cfg.n_gpu = torch.cuda.device_count()
        cfg.local_rank = 0
        cfg.distributed = False
        cfg.batch_sz = cfg.training.batch_size

        if cfg.n_gpu > 1:
            cfg.distributed = True
            cfg.training.batch_size = cfg.batch_sz // cfg.n_gpu

    if cfg.distributed:
        init_distributed(cfg)

    if cfg.local_rank == 0:
        print('CUDA_VISIBLE_DEVICES', os.environ["CUDA_VISIBLE_DEVICES"])

    init_record_setting(cfg)


def seed_setting(cfg: DictConfig) -> None:
    """Seed settings."""
    os.environ["PYTHONHASHSEED"] = str(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def init_distributed(cfg: DictConfig):
    """Initialize distributed mode."""
    with open_dict(cfg):
        cfg.global_rank = int(os.environ['RANK'])
        cfg.local_rank = int(os.environ['LOCAL_RANK'])
        cfg.world_size = int(os.environ['WORLD_SIZE'])

    torch.cuda.set_device(cfg.local_rank)

    dist.init_process_group(
        backend="nccl",
        init_method=cfg.init_method,
        world_size=cfg.world_size,
        rank=cfg.local_rank
    )
    dist.barrier()


def init_record_setting(cfg: DictConfig) -> None:
    """Initialize experiment record settings."""
    setting = '{}_{}_ST{}S{}_E{}H{}_S{}H{}F{}_T{}H{}F{}_S{}H{}F{}_D{}_LR{}~{}_D{}_E{}B{}'.format(
        cfg.model.name,
        cfg.dataset.name,
        cfg.model.n_st_blocks,
        cfg.model.n_spat_blocks,
        cfg.model.embed_dim,
        cfg.model.hid_dim,
        cfg.model.spec_dmodel,
        cfg.model.spec_nheads,
        cfg.model.spec_dim_factor,
        cfg.model.temp_dmodel,
        cfg.model.temp_nheads,
        cfg.model.temp_dim_factor,
        cfg.model.spat_dmodel,
        cfg.model.spat_nheads,
        cfg.model.spat_dim_factor,
        cfg.model.dropout,
        cfg.optimizer.lr,
        cfg.optimizer.lr_scheduler.target_rate,
        cfg.optimizer.weight_decay,
        cfg.training.train_epochs,
        cfg.batch_sz,
    )
    unique_id = '{}_{}'.format(cfg.unique_id, setting)
    path = os.path.join(cfg.checkpoints, unique_id)

    with open_dict(cfg):
        cfg.unique_id = unique_id
        cfg.path = path

    if cfg.local_rank == 0:
        os.makedirs(path, exist_ok=True)
        print('Experiment ID: ', unique_id)
