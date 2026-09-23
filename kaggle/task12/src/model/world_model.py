import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.rssm import RSSM


class WorldModel(nn.Module):
    """
    Mini-DreamerV3 world model (XS config).
    Encoder: obs -> embed (MLP, 1 layer)
    RSSM: (h, z, a, embed) -> (h', z', prior_logits, post_logits)
    Decoder: [h, z] -> obs (MLP, 1 hidden layer)
    """

    def __init__(self, obs_dim, act_dim, cfg):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        deter   = cfg['rssm_deter']
        stoch   = cfg['rssm_stoch']
        classes = cfg['rssm_classes']
        hidden  = cfg['rssm_hidden']
        embed   = cfg['embed_dim']

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, embed),
            nn.ELU(),
        )

        self.rssm = RSSM(
            deter=deter,
            stoch=stoch,
            classes=classes,
            hidden=hidden,
            embed_dim=embed,
            act_dim=act_dim,
        )

        self.decoder = nn.Sequential(
            nn.Linear(deter + stoch * classes, hidden),
            nn.ELU(),
            nn.Linear(hidden, obs_dim),
        )

    def forward_sequence(self, obs, actions, kl_free=1.0):
        """
        Process a sequence of observations and actions.
        obs:     (B, T, obs_dim)
        actions: (B, T, act_dim)  -- actions[t] taken after obs[t]

        Returns per-step dicts with h, z, prior_logits, post_logits,
        decoded_obs, kl (all tensors, first dim = B*T after reshape).
        """
        B, T, _ = obs.shape
        device = obs.device

        h, z = self.rssm.initial_state(B, device)

        all_h, all_z, all_prior, all_post = [], [], [], []
        all_decoded, all_kl, all_recon = [], [], []

        for t in range(T):
            obs_t   = obs[:, t]
            prev_a  = actions[:, t - 1] if t > 0 else torch.zeros(B, self.act_dim, device=device)

            embed = self.encoder(obs_t)
            h, z, prior_logits, post_logits = self.rssm.observe_step(h, z, prev_a, embed)

            decoded = self.decoder(torch.cat([h, z], dim=-1))

            kl   = self.rssm.kl_divergence(post_logits, prior_logits, free_bits=kl_free)
            recon = F.mse_loss(decoded, obs_t, reduction='none').sum(dim=-1)  # (B,)

            all_h.append(h.detach())
            all_z.append(post_logits.detach())    # save posterior logits as z_t
            all_prior.append(prior_logits.detach())
            all_post.append(post_logits.detach())
            all_decoded.append(decoded.detach())
            all_kl.append(kl.detach())
            all_recon.append(recon.detach())

        return {
            'h':         torch.stack(all_h, dim=1),        # (B, T, deter)
            'z':         torch.stack(all_z, dim=1),        # (B, T, z_dim)
            'prior':     torch.stack(all_prior, dim=1),    # (B, T, z_dim)
            'post':      torch.stack(all_post, dim=1),     # (B, T, z_dim)
            'decoded':   torch.stack(all_decoded, dim=1),  # (B, T, obs_dim)
            'kl':        torch.stack(all_kl, dim=1),       # (B, T)
            'recon':     torch.stack(all_recon, dim=1),    # (B, T)
        }

    def compute_loss(self, obs, actions, kl_free=1.0, kl_scale=1.0):
        """Full training loss on a sequence batch."""
        B, T, _ = obs.shape
        device = obs.device

        h, z = self.rssm.initial_state(B, device)
        total_recon = torch.tensor(0.0, device=device)
        total_kl    = torch.tensor(0.0, device=device)

        for t in range(T):
            obs_t  = obs[:, t]
            prev_a = actions[:, t - 1] if t > 0 else torch.zeros(B, self.act_dim, device=device)

            embed = self.encoder(obs_t)
            h, z, prior_logits, post_logits = self.rssm.observe_step(h, z, prev_a, embed)

            decoded = self.decoder(torch.cat([h, z], dim=-1))

            recon_loss = F.mse_loss(decoded, obs_t)
            kl         = self.rssm.kl_divergence(post_logits, prior_logits, free_bits=kl_free).mean()

            total_recon = total_recon + recon_loss
            total_kl    = total_kl + kl

        loss = total_recon / T + kl_scale * (total_kl / T)
        return loss, (total_recon / T).item(), (total_kl / T).item()

    @torch.no_grad()
    def infer_sequence(self, obs, actions):
        """Run model in eval mode, return per-step states and signals (no grad)."""
        was_training = self.training
        self.eval()

        B, T, _ = obs.shape
        device = obs.device

        h, z = self.rssm.initial_state(B, device)

        h_list, z_list, kl_list, recon_list = [], [], [], []

        for t in range(T):
            obs_t  = obs[:, t]
            prev_a = actions[:, t - 1] if t > 0 else torch.zeros(B, self.act_dim, device=device)

            embed = self.encoder(obs_t)
            h, z, prior_logits, post_logits = self.rssm.observe_step(h, z, prev_a, embed)
            decoded = self.decoder(torch.cat([h, z], dim=-1))

            kl   = self.rssm.kl_divergence(post_logits, prior_logits, free_bits=0.0)
            recon = F.mse_loss(decoded, obs_t, reduction='none').sum(dim=-1)

            h_list.append(h)
            z_list.append(post_logits)
            kl_list.append(kl)
            recon_list.append(recon)

        if was_training:
            self.train()

        return {
            'h':    torch.stack(h_list, dim=1),    # (B, T, deter)
            'z':    torch.stack(z_list, dim=1),    # (B, T, z_dim)
            'kl':   torch.stack(kl_list, dim=1),   # (B, T)
            'recon': torch.stack(recon_list, dim=1), # (B, T)
        }

    @torch.no_grad()
    def rollout_variance(self, h_batch, n_samples=5, horizon=5):
        """
        Probe B signal: variance of imagined observations from h_t.
        h_batch: (N, deter)
        Returns variance of decoded obs across samples: (N,)
        """
        self.eval()
        N = h_batch.shape[0]
        device = h_batch.device

        decoded_samples = []
        for _ in range(n_samples):
            h = h_batch.clone()
            z = torch.zeros(N, self.rssm.z_dim, device=device)
            a = torch.zeros(N, self.act_dim, device=device)

            obs_trajectory = []
            for _ in range(horizon):
                h, z, _ = self.rssm.imagine_step(h, z, a)
                dec = self.decoder(torch.cat([h, z], dim=-1))
                obs_trajectory.append(dec)

            # Use the final predicted observation
            decoded_samples.append(obs_trajectory[-1])

        # Variance across samples: (N, obs_dim) -> mean over obs_dim -> (N,)
        stacked = torch.stack(decoded_samples, dim=0)   # (n_samples, N, obs_dim)
        var = stacked.var(dim=0).mean(dim=-1)            # (N,)
        return var
