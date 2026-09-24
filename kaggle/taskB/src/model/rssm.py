import torch
import torch.nn as nn
import torch.nn.functional as F


class RSSM(nn.Module):
    """
    Recurrent State Space Model (DreamerV3 architecture).
    Deterministic state h_t (GRU hidden), stochastic state z_t (categorical).

    Sequence at each step t:
      h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])
      prior:  p(z_t | h_t)           -- imagination, no observation
      posterior: q(z_t | h_t, e_t)   -- observation update, e_t = encode(x_t)
    """

    def __init__(self, deter=256, stoch=32, classes=32, hidden=256, embed_dim=64, act_dim=1):
        super().__init__()
        self.deter = deter
        self.stoch = stoch
        self.classes = classes
        self.z_dim = stoch * classes   # 1024 for XS

        self.gru = nn.GRUCell(self.z_dim + act_dim, deter)

        # Prior: h_t -> z logits
        self.prior_net = nn.Sequential(
            nn.Linear(deter, hidden),
            nn.ELU(),
            nn.Linear(hidden, self.z_dim),
        )

        # Posterior: [h_t, embed] -> z logits
        self.post_net = nn.Sequential(
            nn.Linear(deter + embed_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, self.z_dim),
        )

    def initial_state(self, batch_size, device='cpu'):
        h = torch.zeros(batch_size, self.deter, device=device)
        z = torch.zeros(batch_size, self.z_dim, device=device)
        return h, z

    def observe_step(self, h, z, action, embed):
        """One RSSM step with observation.
        Returns h_t, z_t (st sample), prior_logits, post_logits.
        """
        inp = torch.cat([z, action], dim=-1)
        h_next = self.gru(inp, h)

        prior_logits = self.prior_net(h_next)
        post_logits = self.post_net(torch.cat([h_next, embed], dim=-1))

        z_next = self._straight_through_sample(post_logits)
        return h_next, z_next, prior_logits, post_logits

    def imagine_step(self, h, z, action):
        """One RSSM step without observation (prior only)."""
        inp = torch.cat([z, action], dim=-1)
        h_next = self.gru(inp, h)

        prior_logits = self.prior_net(h_next)
        z_next = self._straight_through_sample(prior_logits)
        return h_next, z_next, prior_logits

    def _straight_through_sample(self, logits):
        """Straight-through categorical: one-hot forward, softmax backward."""
        B = logits.shape[0]
        logits_rs = logits.view(B, self.stoch, self.classes)

        # Hard sample (one-hot)
        indices = torch.distributions.Categorical(logits=logits_rs).sample()
        z_hard = F.one_hot(indices, self.classes).float()

        # Soft (differentiable)
        z_soft = torch.softmax(logits_rs, dim=-1)

        # Straight-through: forward=hard, backward=soft
        z_st = z_hard + (z_soft - z_soft.detach())
        return z_st.view(B, self.z_dim)

    def kl_divergence(self, post_logits, prior_logits, free_bits=0.0):
        """Per-sample KL(posterior || prior), summed over stoch dim.
        Returns shape (batch,).
        """
        B = post_logits.shape[0]
        post = post_logits.view(B, self.stoch, self.classes)
        prior = prior_logits.view(B, self.stoch, self.classes)

        log_post = F.log_softmax(post, dim=-1)
        log_prior = F.log_softmax(prior, dim=-1)
        post_prob = torch.softmax(post, dim=-1)

        # KL for each categorical: sum over classes, then sum over stoch dims
        kl_per_cat = (post_prob * (log_post - log_prior)).sum(dim=-1)  # (B, stoch)
        kl = kl_per_cat.sum(dim=-1)                                     # (B,)

        if free_bits > 0.0:
            kl = torch.clamp(kl, min=free_bits * self.stoch)

        return kl
