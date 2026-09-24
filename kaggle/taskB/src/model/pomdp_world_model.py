"""
PomdpWorldModel — Section 10's controlled full-vs-partial-observability variant.

Deliberately a SEPARATE class from src.model.world_model.WorldModel rather than a
modification of it: WorldModel's obs_dim is shared by encoder input and decoder
output/target everywhere (forward_sequence, compute_loss, infer_sequence,
rollout_variance), and every other phase's frozen checkpoints and analysis code
depend on that invariant holding. Touching it risks subtly breaking Phases 0-3.

This class reuses RSSM as-is (it only consumes `embed`/`action`, agnostic to what
produced them) but wires:
  encoder: MASKED observation (obs_dim_in, e.g. position-only, 3-dim for cartpole)
  decoder: FULL state target (obs_dim_out, e.g. position+velocity, 5-dim)
so the model is forced to infer the hidden (masked) components from h_t/z_t --
the genuine POMDP/filtering setup Section 10 and Section 12's theory chapter
describe, rather than a trivial "reconstruct what you can already see" task.

Architecture, hyperparameters, and everything else (RSSM deter/stoch/classes,
optimizer, seq_len, batch, lr, kl_free, kl_scale) are IDENTICAL to the existing
XS_CONFIG cartpole model, per Section 10.1's "keep as much as possible unchanged."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.rssm import RSSM


class PomdpWorldModel(nn.Module):
    def __init__(self, obs_dim_in, obs_dim_out, act_dim, cfg):
        super().__init__()
        self.obs_dim_in = obs_dim_in
        self.obs_dim_out = obs_dim_out
        self.act_dim = act_dim

        deter   = cfg['rssm_deter']
        stoch   = cfg['rssm_stoch']
        classes = cfg['rssm_classes']
        hidden  = cfg['rssm_hidden']
        embed   = cfg['embed_dim']

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim_in, embed),
            nn.ELU(),
        )
        self.rssm = RSSM(deter=deter, stoch=stoch, classes=classes,
                          hidden=hidden, embed_dim=embed, act_dim=act_dim)
        self.decoder = nn.Sequential(
            nn.Linear(deter + stoch * classes, hidden),
            nn.ELU(),
            nn.Linear(hidden, obs_dim_out),
        )

    def compute_loss(self, obs_in, obs_target, actions, kl_free=1.0, kl_scale=1.0):
        """obs_in: (B,T,obs_dim_in) masked/observed. obs_target: (B,T,obs_dim_out)
        full state, the reconstruction target. actions: (B,T,act_dim)."""
        B, T, _ = obs_in.shape
        device = obs_in.device
        h, z = self.rssm.initial_state(B, device)
        total_recon = torch.tensor(0.0, device=device)
        total_kl = torch.tensor(0.0, device=device)

        for t in range(T):
            obs_t = obs_in[:, t]
            target_t = obs_target[:, t]
            prev_a = actions[:, t - 1] if t > 0 else torch.zeros(B, self.act_dim, device=device)

            embed = self.encoder(obs_t)
            h, z, prior_logits, post_logits = self.rssm.observe_step(h, z, prev_a, embed)
            decoded = self.decoder(torch.cat([h, z], dim=-1))

            recon_loss = F.mse_loss(decoded, target_t)
            kl = self.rssm.kl_divergence(post_logits, prior_logits, free_bits=kl_free).mean()

            total_recon = total_recon + recon_loss
            total_kl = total_kl + kl

        loss = total_recon / T + kl_scale * (total_kl / T)
        return loss, (total_recon / T).item(), (total_kl / T).item()

    @torch.no_grad()
    def infer_sequence(self, obs_in, obs_target, actions):
        was_training = self.training
        self.eval()
        B, T, _ = obs_in.shape
        device = obs_in.device
        h, z = self.rssm.initial_state(B, device)
        h_list, z_list, kl_list, recon_list = [], [], [], []
        for t in range(T):
            obs_t = obs_in[:, t]
            target_t = obs_target[:, t]
            prev_a = actions[:, t - 1] if t > 0 else torch.zeros(B, self.act_dim, device=device)
            embed = self.encoder(obs_t)
            h, z, prior_logits, post_logits = self.rssm.observe_step(h, z, prev_a, embed)
            decoded = self.decoder(torch.cat([h, z], dim=-1))
            kl = self.rssm.kl_divergence(post_logits, prior_logits, free_bits=0.0)
            recon = F.mse_loss(decoded, target_t, reduction='none').sum(dim=-1)
            h_list.append(h); z_list.append(post_logits); kl_list.append(kl); recon_list.append(recon)
        if was_training:
            self.train()
        return dict(h=torch.stack(h_list, dim=1), z=torch.stack(z_list, dim=1),
                    kl=torch.stack(kl_list, dim=1), recon=torch.stack(recon_list, dim=1))
