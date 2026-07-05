"""
Generalised dm_control wrapper (domain/task parametrised) for Task D.

Mirrors CartpoleEnv's interface but works for any dm_control domain, so the
whole Phase-1 pipeline can be re-run on a structurally different environment
(reacher, pendulum, acrobot, …) without touching the model or probe code.
"""

import numpy as np
from dm_control import suite


class DMCEnv:
    def __init__(self, domain='reacher', task='easy', noisy=False, noise_std=0.1, seed=None):
        self.domain = domain
        self.task = task
        self.noisy = noisy
        self.noise_std = noise_std
        self._rng = np.random.default_rng(seed)
        self._env = suite.load(domain, task, task_kwargs={'random': seed or 0})
        ts = self._env.reset()
        self.obs_dim = int(np.concatenate([v.flatten() for v in ts.observation.values()]).shape[0])
        self.act_dim = int(self._env.action_spec().shape[0])

    def reset(self):
        return self._process_obs(self._env.reset())

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
        ts = self._env.step(action)
        obs = self._process_obs(ts)
        done = ts.last()
        rew = float(ts.reward) if ts.reward is not None else 0.0
        return obs, rew, done

    def _process_obs(self, ts):
        flat = np.concatenate([v.flatten() for v in ts.observation.values()]).astype(np.float32)
        if self.noisy:
            flat = flat + self._rng.standard_normal(flat.shape).astype(np.float32) * self.noise_std
        return flat
