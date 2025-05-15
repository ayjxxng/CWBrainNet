from typing import List, Tuple, Optional, Dict, Union
import torch
from torch import nn, cfloat
import torch.nn.functional as F
from omegaconf import DictConfig
from models.modules import Res2DModule, ComplexLinear, EncoderLayer, CrossEncoderLayer


class Encoder(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout: float, num_layers: int) -> None:
        super().__init__()
        self.layers_real = nn.ModuleList([
            EncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True, norm_first=True
            ) for _ in range(num_layers)
        ])
        self.layers_imag = nn.ModuleList([
            EncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True, norm_first=True
            ) for _ in range(num_layers)
        ])

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        _shape = emb.shape
        emb = emb.flatten(0, 1)

        emb_r, emb_i = emb.real, emb.imag
        for layer_real, layer_imag in zip(self.layers_real, self.layers_imag):
            emb_r = layer_real(emb_r)
            emb_i = layer_imag(emb_i)

        emb_r = emb_r + emb.real
        emb_i = emb_i + emb.imag

        return torch.complex(emb_r, emb_i).view(_shape)


class SpatialEnc(nn.Module):
    def __init__(self, cfg: DictConfig, flag: str = 'self') -> None:
        super().__init__()
        self.cfg = cfg.model
        self.flag = flag

        self.linear_in = ComplexLinear(self.cfg.embed_dim * 2, self.cfg.spat_dmodel)
        _module = EncoderLayer if flag == 'self' else CrossEncoderLayer
        self.layers_real = nn.ModuleList([
            _module(
                d_model=self.cfg.spat_dmodel, nhead=self.cfg.spat_nheads,
                dim_feedforward=self.cfg.spat_dmodel * self.cfg.spat_dim_factor,
                dropout=self.cfg.dropout, batch_first=True, norm_first=True,
            ) for _ in range(self.cfg.n_spat_blocks)
        ])
        self.layers_imag = nn.ModuleList([
            _module(
                d_model=self.cfg.spat_dmodel, nhead=self.cfg.spat_nheads,
                dim_feedforward=self.cfg.spat_dmodel * self.cfg.spat_dim_factor,
                dropout=self.cfg.dropout, batch_first=True, norm_first=True,
            ) for _ in range(self.cfg.n_spat_blocks)
        ])

    def forward(self, batch: Dict[str, torch.tensor]) -> Dict[str, torch.tensor]:
        """
        Args:
            - 'spat_emb': Complex-valued spatial embedding, shape [B, R+1, 2E]

        Returns:
            - 'spat_self' or 'spat_cross': Refined spatial embedding, shape [B, R+1, ds]

        """
        emb = self.linear_in(batch['spat_emb']) # [B, R+1, ds]

        if self.flag == 'cross':
            emb_r, emb_i = emb.imag, emb.real
            for layer_real, layer_imag in zip(self.layers_real, self.layers_imag):
                emb_r = layer_real(emb_r, emb.real, emb.real)
                emb_i = layer_imag(emb_i, emb.imag, emb.imag)
        else:
            emb_r, emb_i = emb.real, emb.imag
            for layer_real, layer_imag in zip(self.layers_real, self.layers_imag):
                emb_r = layer_real(emb_r)
                emb_i = layer_imag(emb_i)

        emb_r = emb_r + emb.real
        emb_i = emb_i + emb.imag

        batch[f'spat_{self.flag}'] = torch.complex(emb_r, emb_i)
        return batch


class ST_Block(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg.model
        self.E = cfg.model.embed_dim
        self.F = cfg.dataset.n_frequencies
        self.R = cfg.dataset.n_channels
        self.T = cfg.dataset.n_times

        # Representation tokens
        self.frt_r = nn.Parameter(torch.rand((1, self.R, 1, self.T)))
        self.frt_i = nn.Parameter(torch.rand((1, self.R, 1, self.T)))

        self.trt_r = nn.Parameter(torch.rand((1, self.R, 1, self.F)))
        self.trt_i = nn.Parameter(torch.rand((1, self.R, 1, self.F)))

        # Embedding layers
        self.temporal_emb = ComplexLinear(self.F, self.cfg.temp_dmodel)
        self.spectral_emb = ComplexLinear(self.T, self.cfg.spec_dmodel)

        # Positional encodings
        self.fpe_r = nn.Parameter(torch.zeros((1, 1, self.F + 1, self.cfg.spec_dmodel)))
        self.fpe_i = nn.Parameter(torch.zeros((1, 1, self.F + 1, self.cfg.spec_dmodel)))

        self.tpe_r = nn.Parameter(torch.zeros((1, 1, self.T + 1, self.cfg.temp_dmodel)))
        self.tpe_i = nn.Parameter(torch.zeros((1, 1, self.T + 1, self.cfg.temp_dmodel)))

        # Spatial tokens
        self.fst_r = nn.Parameter(torch.zeros((1, 1, 1, self.cfg.spec_dmodel)))
        self.fst_i = nn.Parameter(torch.zeros((1, 1, 1, self.cfg.spec_dmodel)))

        self.tst_r = nn.Parameter(torch.zeros((1, 1, 1, self.cfg.temp_dmodel)))
        self.tst_i = nn.Parameter(torch.zeros((1, 1, 1, self.cfg.temp_dmodel)))

        # Encoder blocks
        self.temporal_blocks = Encoder(
            self.cfg.temp_dmodel, self.cfg.temp_nheads,
            self.cfg.temp_dmodel * self.cfg.temp_dim_factor,
            self.cfg.dropout, self.cfg.n_st_blocks
        )
        self.spectral_blocks = Encoder(
            self.cfg.spec_dmodel, self.cfg.spec_nheads,
            self.cfg.spec_dmodel * self.cfg.spec_dim_factor,
            self.cfg.dropout, self.cfg.n_st_blocks
        )

        # Projection layers
        self.proj_time = ComplexLinear(self.cfg.temp_dmodel, self.E)
        self.proj_freq = ComplexLinear(self.cfg.spec_dmodel, self.E)

    def forward(self, batch: Dict[str, torch.tensor]):
        """
        Args:
            - 'embeds': Complex-valued input after CWT embedding, shape [B, R, F, T]

        Returns:
            - 'spat_emb': Integrated embedding, shape [B, R+1, 2E]
        """
        B, device = batch['embeds'].size(0), batch['embeds'].device

        # Spectral path
        emb_spec = batch['embeds'].clone()
        frt = torch.repeat_interleave(
            torch.complex(self.frt_r, self.frt_i), B, 0
        ).to(device)
        emb_spec = torch.cat([frt, emb_spec], dim=2) # [B, R, F+1, T]

        emb_spec = self.spectral_emb(emb_spec)
        emb_spec += torch.complex(self.fpe_r, self.fpe_i)

        fst = torch.repeat_interleave(
            torch.complex(self.fst_r, self.fst_i), B, 0
        ).to(device)
        fst = nn.functional.pad(fst, (0, 0, 0, self.F))
        emb_spec = torch.cat([fst, emb_spec], dim=1) # [B, R+1, F+1, dt]

        # Temporal path
        emb_temp = batch['embeds'].clone().transpose(-2, -1)
        trt = torch.repeat_interleave(
            torch.complex(self.trt_r, self.trt_i), B, 0
        ).to(device)
        emb_temp = torch.cat([trt, emb_temp], dim=-2) # [B, R, T+1, F]

        emb_temp = self.temporal_emb(emb_temp)
        emb_temp += torch.complex(self.tpe_r, self.tpe_i)

        tst = torch.repeat_interleave(
            torch.complex(self.tst_r, self.tst_i), B, 0
        ).to(device)
        tst = nn.functional.pad(tst, (0, 0, 0, self.T))
        emb_temp = torch.cat([tst, emb_temp], dim=1) # [B, R+1, T+1, df]

        # Apply encoder blocks
        emb_temp = self.temporal_blocks(emb_temp)
        emb_spec = self.spectral_blocks(emb_spec)

        # Project tokens
        proj_temp = self.proj_time(emb_temp[:, :, 0, :])
        proj_spec = self.proj_freq(emb_spec[:, :, 0, :])

        # Concat both projections
        batch['spat_emb'] = torch.cat([proj_temp, proj_spec], dim=-1) # [B, R+1, 2E]

        return batch


class CWBrainNet(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg.model
        self.loss_fn = nn.CrossEntropyLoss(reduction="mean")

        # CWT embedding blocks
        self.CWTEmb_real = Res2DModule(cfg)
        self.CWTEmb_imag = Res2DModule(cfg)

        # ST-Block
        self.ST_Block = ST_Block(cfg)

        # Spatial C-Block
        self.spatial_cross_blocks = SpatialEnc(cfg, 'cross')
        self.spatial_self_blocks = SpatialEnc(cfg, 'self')
        self.linear_out = ComplexLinear(self.cfg.spat_dmodel * 2, self.cfg.hid_dim)

        # Classifier
        self.classifier = nn.Linear(self.cfg.hid_dim * 2, cfg.dataset.n_classes)

    def forward(self, batch: Dict[str, torch.tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            - 'inputs': Complex-valued input after CWT, shape [B, T, F, R, 2]
                        where the last dim is (real, imag)
            - 'labels': Ground-truth class labels, shape [B]

        Returns:
            - 'loss': Cross-entropy loss, shape []
            - 'probs': Probability for class 1, shape [B]
        """

        # Extract real and imaginary components: [B, R, F, T]
        x_r = batch['inputs'][..., 0].permute(0, 3, 2, 1)
        x_i = batch['inputs'][..., 1].permute(0, 3, 2, 1)

        # CWT embedding block
        x_r = self.CWTEmb_real(x_r)
        x_i = self.CWTEmb_imag(x_i)
        batch['embeds'] = torch.complex(x_r, x_i)

        # ST-Block
        batch = self.ST_Block(batch)

        # Spatial C-Block
        batch = self.spatial_cross_blocks(batch)
        batch = self.spatial_self_blocks(batch)
        out = self.linear_out(
            torch.cat([batch['spat_self'][:, 0, :], batch['spat_cross'][:, 0, :]], dim=-1)
        )

        # Classifier
        logits = self.classifier(
            torch.cat([out.real, out.imag], dim=-1)
        )
        return self.classification_loss(logits, batch['labels'].detach())

    def classification_loss(self, logits, labels):
        return {
            'loss': self.loss_fn(logits, labels.float()),
            'probs': F.softmax(logits, dim=-1)[:, 1],
        }
