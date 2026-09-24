import numpy as np
from dm_control import suite


class CartpoleEnv:
    """Thin wrapper around dm_control cartpole tasks."""

    def __init__(self, task='swingup', noisy=False, noise_std=0.1, seed=None):
        self.noisy = noisy
        self.noise_std = noise_std
        self._rng = np.random.default_rng(seed)
        self._env = suite.load('cartpole', task, task_kwargs={'random': seed or 0})
        self.obs_dim = 5    # position(3) + velocity(2)
        self.act_dim = 1

    def reset(self):
        ts = self._env.reset()
        return self._process_obs(ts)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
        ts = self._env.step(action)
        obs  = self._process_obs(ts)
        done = ts.last()
        rew  = float(ts.reward) if ts.reward is not None else 0.0
        return obs, rew, done

    def _process_obs(self, ts):
        obs = ts.observation
        flat = np.concatenate([v.flatten() for v in obs.values()]).astype(np.float32)
        if self.noisy:
            flat = flat + self._rng.standard_normal(flat.shape).astype(np.float32) * self.noise_std
        return flat
