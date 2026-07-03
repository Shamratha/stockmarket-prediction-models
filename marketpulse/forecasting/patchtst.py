"""PatchTST (Nie et al. 2023) — patch the series into tokens, run a
Transformer encoder over patches, flatten into the forecast head."""

import torch
import torch.nn as nn

from .torch_models import TorchForecaster


class PatchTSTNet(nn.Module):
    def __init__(self, window, patch_len=8, stride=8, d_model=64, heads=4,
                 layers=2, dropout=0.1, horizon=1):
        super().__init__()
        self.patch_len, self.stride = patch_len, stride
        n_patches = (window - patch_len) // stride + 1
        self.embed = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(
            d_model, heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(n_patches * d_model, horizon)

    def forward(self, x):
        patches = x.unfold(-1, self.patch_len, self.stride)   # (B, n_patches, patch_len)
        h = self.embed(patches) + self.pos
        h = self.encoder(h)
        return self.head(h.flatten(1))


class PatchTSTForecaster(TorchForecaster):
    name = 'PatchTST'

    def __init__(self, window=64, patch_len=8, stride=8, d_model=64, **kw):
        kw.setdefault('window', window)
        super().__init__(**kw)
        self.patch_len, self.stride, self.d_model = patch_len, stride, d_model

    def build_net(self):
        return PatchTSTNet(self.window, self.patch_len, self.stride, self.d_model, horizon=1)
