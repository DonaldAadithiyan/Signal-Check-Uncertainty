import random
from collections import deque
import numpy as np
import torch


class EpisodeReplayBuffer:
    """Stores complete episodes; samples random fixed-length sequences."""

    def __init__(self, capacity=500):
        self.capacity = capacity
        self._episodes: deque = deque()
        self._total_steps = 0

    def add_episode(self, obs_list: list, action_list: list):
        if len(action_list) < 2:
            return
        ep = {
            'obs':     np.array(obs_list,    dtype=np.float32),    # (T, obs_dim)
            'actions': np.array(action_list, dtype=np.float32),    # (T, act_dim)
        }
        self._episodes.append(ep)
        self._total_steps += len(action_list)
        while len(self._episodes) > self.capacity:
            old = self._episodes.popleft()
            self._total_steps -= len(old['actions'])

    def __len__(self):
        return self._total_steps

    def sample(self, batch_size: int, seq_len: int, device: str = 'cpu'):
        """Returns obs (B, T, D) and actions (B, T, A) tensors."""
        obs_seqs, act_seqs = [], []
        valid = [ep for ep in self._episodes if len(ep['actions']) >= seq_len]
        if not valid:
            valid = list(self._episodes)

        for _ in range(batch_size):
            ep = random.choice(valid)
            T  = len(ep['actions'])
            if T >= seq_len:
                start = random.randint(0, T - seq_len)
                obs_seqs.append(ep['obs'][start:start + seq_len])
                act_seqs.append(ep['actions'][start:start + seq_len])
            else:
                # Pad short episode
                pad_obs = np.zeros((seq_len, ep['obs'].shape[1]),     dtype=np.float32)
                pad_act = np.zeros((seq_len, ep['actions'].shape[1]), dtype=np.float32)
                pad_obs[:T] = ep['obs']
                pad_act[:T] = ep['actions']
                obs_seqs.append(pad_obs)
                act_seqs.append(pad_act)

        obs_t = torch.tensor(np.stack(obs_seqs), dtype=torch.float32, device=device)
        act_t = torch.tensor(np.stack(act_seqs), dtype=torch.float32, device=device)
        return obs_t, act_t
