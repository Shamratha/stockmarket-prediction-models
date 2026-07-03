"""PyTorch sequence forecasters: LSTM, GRU, Transformer.

All share one recipe: a window of the last `window` log returns in, the next
log return out. Trained with AdamW + early stopping on a validation tail.
"""

import numpy as np
import torch
import torch.nn as nn

from .base import Forecaster

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def make_windows(returns, window):
    X, y = [], []
    for i in range(window, len(returns)):
        X.append(returns[i - window:i])
        y.append(returns[i])
    return (
        torch.tensor(np.array(X), dtype=torch.float32),
        torch.tensor(np.array(y), dtype=torch.float32),
    )


class TorchForecaster(Forecaster):
    """Shared training / prediction loop; subclasses provide the network."""

    def __init__(self, window=32, epochs=200, lr=1e-3, batch_size=64, patience=20, seed=42):
        self.window = window
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.seed = seed
        self.net = None
        self.mu = 0.0
        self.sigma = 1.0

    def build_net(self):
        raise NotImplementedError

    def fit(self, train_returns):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        r = np.asarray(train_returns, dtype=np.float32)
        self.mu, self.sigma = float(r.mean()), float(r.std() + 1e-8)
        z = (r - self.mu) / self.sigma

        X, y = make_windows(z, self.window)
        n_val = max(32, int(0.15 * len(X)))
        X_tr, y_tr, X_val, y_val = X[:-n_val], y[:-n_val], X[-n_val:], y[-n_val:]

        self.net = self.build_net().to(DEVICE)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        loss_fn = nn.MSELoss()

        best_val, best_state, bad = np.inf, None, 0
        for epoch in range(self.epochs):
            self.net.train()
            perm = torch.randperm(len(X_tr))
            for i in range(0, len(X_tr), self.batch_size):
                idx = perm[i:i + self.batch_size]
                xb, yb = X_tr[idx].to(DEVICE), y_tr[idx].to(DEVICE)
                opt.zero_grad()
                pred = self.net(xb).squeeze(-1)
                loss = loss_fn(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()

            self.net.eval()
            with torch.no_grad():
                val = loss_fn(self.net(X_val.to(DEVICE)).squeeze(-1), y_val.to(DEVICE)).item()
            if val < best_val - 1e-6:
                best_val, bad = val, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()

    def _z(self, arr):
        return (np.asarray(arr, dtype=np.float32) - self.mu) / self.sigma

    def predict_history(self, full_returns, test_size):
        z = self._z(full_returns)
        X = []
        for t in range(len(z) - test_size, len(z)):
            X.append(z[t - self.window:t])
        with torch.no_grad():
            pred = self.net(torch.tensor(np.array(X), dtype=torch.float32).to(DEVICE))
        return pred.squeeze(-1).cpu().numpy() * self.sigma + self.mu

    def _predict_one(self, history):
        z = self._z(history)[-self.window:]
        with torch.no_grad():
            pred = self.net(torch.tensor(z[None, :], dtype=torch.float32).to(DEVICE))
        return float(pred.squeeze()) * self.sigma + self.mu


class LSTMForecaster(TorchForecaster):
    name = 'LSTM'

    def __init__(self, hidden=64, layers=2, dropout=0.1, **kw):
        super().__init__(**kw)
        self.hidden, self.layers, self.dropout = hidden, layers, dropout

    def build_net(self):
        window, hidden, layers, dropout = self.window, self.hidden, self.layers, self.dropout

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.rnn = nn.LSTM(1, hidden, layers, batch_first=True,
                                   dropout=dropout if layers > 1 else 0.0)
                self.head = nn.Linear(hidden, 1)

            def forward(self, x):
                out, _ = self.rnn(x.unsqueeze(-1))
                return self.head(out[:, -1])

        return Net()


class GRUForecaster(LSTMForecaster):
    name = 'GRU'

    def build_net(self):
        window, hidden, layers, dropout = self.window, self.hidden, self.layers, self.dropout

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.rnn = nn.GRU(1, hidden, layers, batch_first=True,
                                  dropout=dropout if layers > 1 else 0.0)
                self.head = nn.Linear(hidden, 1)

            def forward(self, x):
                out, _ = self.rnn(x.unsqueeze(-1))
                return self.head(out[:, -1])

        return Net()


class TransformerForecaster(TorchForecaster):
    name = 'Transformer'

    def __init__(self, d_model=64, heads=4, layers=2, dropout=0.1, **kw):
        super().__init__(**kw)
        self.d_model, self.heads, self.n_layers, self.dropout = d_model, heads, layers, dropout

    def build_net(self):
        window, d_model, heads, n_layers, dropout = (
            self.window, self.d_model, self.heads, self.n_layers, self.dropout
        )

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Linear(1, d_model)
                self.pos = nn.Parameter(torch.randn(1, window, d_model) * 0.02)
                enc = nn.TransformerEncoderLayer(
                    d_model, heads, dim_feedforward=d_model * 4,
                    dropout=dropout, batch_first=True, norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(enc, n_layers)
                self.head = nn.Linear(d_model, 1)

            def forward(self, x):
                h = self.embed(x.unsqueeze(-1)) + self.pos
                h = self.encoder(h)
                return self.head(h[:, -1])

        return Net()
