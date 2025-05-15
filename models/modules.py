import math
import numbers
import warnings
from typing import List, Optional, Tuple, Union

import torch
from torch import nn, Tensor, Size
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from omegaconf import DictConfig


class Res2DModule(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.n_channels = cfg.dataset.n_channels
        self.batch_norm = cfg.model.batch_norm

        self.conv_1 = nn.Conv2d(self.n_channels, self.n_channels, (1, 3), padding='same')
        self.conv_2 = nn.Conv2d(self.n_channels, self.n_channels, (1, 3), padding='same')
        self.activation = nn.SiLU()

        if self.batch_norm:
            self.bn_1 = nn.BatchNorm2d(self.n_channels)
            self.bn_2 = nn.BatchNorm2d(self.n_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_1(x)
        if self.batch_norm:
            out = self.bn_1(out)
        out = self.conv_2(self.activation(out))
        if self.batch_norm:
            out = self.bn_2(out)
        return x + out


class ComplexLinear(nn.Module):
    """
    This module is different from a typical complex-valued linear layer,
    as it performs linear operations on the real and imaginary components independently.
    """
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear_real = nn.Linear(in_dim, out_dim)
        self.linear_imag = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_r = self.linear_real(x.real)
        out_i = self.linear_imag(x.imag)
        return torch.complex(out_r, out_i)


class RMSNorm(nn.Module):
    def __init__(self, normalized_shape: Union[int, List[int], Size], eps: float = 1e-5,
                 bias: bool = False, device=None, dtype=None) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        self.bias = Parameter(torch.empty(self.normalized_shape)) if bias else None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        x_norm = x * torch.rsqrt(var)

        rmsnorm = self.weight * x_norm

        if self.bias is not None:
            rmsnorm = rmsnorm + self.bias

        return rmsnorm


class EncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation=nn.SiLU(), layer_norm_eps: float = 1e-5, batch_first: bool = True,
                 norm_first: bool = True, bias: bool = True, device=None,  dtype=None) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation,
                         layer_norm_eps, batch_first, norm_first, device, dtype)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                               bias=bias, batch_first=batch_first,
                                               **factory_kwargs)
        self.norm1 = RMSNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = RMSNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.activation = activation
        self.attention_weights: Optional[Tensor] = None

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, "activation"):
            self.activation = nn.SiLU()

    def forward(self, x, is_causal: bool = False):
        x = x + self._sa_block(self.norm1(x))
        x = x + self._ff_block(self.norm2(x))
        return x

    def _sa_block(self, x: Tensor, attn_mask=None,
                  key_padding_mask=None, is_causal=None) -> Tensor:
        x, weights = self.self_attn(x, x, x,
                                    attn_mask=attn_mask,
                                    key_padding_mask=key_padding_mask,
                                    need_weights=True,
                                    average_attn_weights=False)
        self.attention_weights = weights
        return self.dropout1(x)

    def get_attention_weights(self):
        return self.attention_weights


class CrossEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation=nn.SiLU(), layer_norm_eps: float = 1e-5, batch_first: bool = True,
                 norm_first: bool = True, bias: bool = True, device=None,  dtype=None) -> None:
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation,
                         layer_norm_eps, batch_first, norm_first, device, dtype)
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                               bias=bias, batch_first=batch_first,
                                               **factory_kwargs)
        self.norm1 = RMSNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = RMSNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.activation = activation
        self.attention_weights: Optional[Tensor] = None

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, "activation"):
            self.activation = nn.SiLU()

    def forward(self, q, k, v, is_causal: bool = False):
        q, k, v = map(self.norm1, (q, k, v))
        x = v + self._sa_block(q, k, v)
        x = x + self._ff_block(self.norm2(x))
        return x

    def _sa_block(self, q, k, v, attn_mask=None,
                  key_padding_mask=None, is_causal: bool = False) -> Tensor:
        x, weights = self.self_attn(q, k, v,
                                    attn_mask=attn_mask,
                                    key_padding_mask=key_padding_mask,
                                    need_weights=True,
                                    average_attn_weights=False)
        self.attention_weights = weights
        return self.dropout1(x)

    def get_attention_weights(self):
        return self.attention_weights
