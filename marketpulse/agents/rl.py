"""DQN and PPO trading agents via stable-baselines3."""

import numpy as np

from .env import TradingEnv


def train_rl_agent(algo, train_prices, window=30, timesteps=100_000,
                   fee_bps=10.0, seed=42, verbose=0):
    from stable_baselines3 import DQN, PPO

    env = TradingEnv(train_prices, window=window, fee_bps=fee_bps)
    if algo == 'DQN':
        model = DQN('MlpPolicy', env, seed=seed, verbose=verbose,
                    learning_rate=1e-4, buffer_size=50_000,
                    exploration_fraction=0.2, target_update_interval=500)
    elif algo == 'PPO':
        model = PPO('MlpPolicy', env, seed=seed, verbose=verbose,
                    learning_rate=3e-4, n_steps=512, batch_size=128,
                    ent_coef=0.01)
    else:
        raise ValueError(algo)
    model.learn(total_timesteps=timesteps, progress_bar=False)
    return model


def rl_actions(model, prices, window=30, fee_bps=10.0):
    """Deterministic rollout of a trained SB3 model over a price series."""
    env = TradingEnv(prices, window=window, fee_bps=fee_bps)
    obs, _ = env.reset()
    actions = np.zeros(len(prices), dtype=int)
    done = False
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        actions[env.t] = int(a)
        obs, _, done, _, _ = env.step(int(a))
    return actions
