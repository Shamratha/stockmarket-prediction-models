"""N-BEATS (Oreshkin et al. 2020) — generic architecture, compact version.

Stacked fully-connected blocks; each block predicts a backcast (subtracted
from the residual input) and a forecast (summed across blocks).
"""

import torch
import torch.nn as nn

from .torch_models import TorchForecaster


class NBeatsBlock(nn.Module):
    def __init__(self, window, hidden, horizon):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(window, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.backcast = nn.Linear(hidden, window)
        self.forecast = nn.Linear(hidden, horizon)

    def forward(self, x):
        h = self.fc(x)
        return self.backcast(h), self.forecast(h)


class NBeatsNet(nn.Module):
    def __init__(self, window, hidden=128, blocks=6, horizon=1):
        super().__init__()
        self.blocks = nn.ModuleList(
            [NBeatsBlock(window, hidden, horizon) for _ in range(blocks)]
        )

    def forward(self, x):
        residual = x
        forecast = 0.0
        for block in self.blocks:
            back, fore = block(residual)
            residual = residual - back
            forecast = forecast + fore
        return forecast


class NBeatsForecaster(TorchForecaster):
    name = 'N-BEATS'

    def __init__(self, hidden=128, blocks=6, window=64, **kw):
        kw.setdefault('window', window)
        super().__init__(**kw)
        self.hidden, self.blocks = hidden, blocks

    def build_net(self):
        return NBeatsNet(self.window, self.hidden, self.blocks, horizon=1)
