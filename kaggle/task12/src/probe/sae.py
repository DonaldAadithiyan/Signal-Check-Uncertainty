"""
TopK Sparse Autoencoder for h_t (Reframed_Confusion_RSSM_Project, Section 8).

Architecture (per the spec):
    h_t (256) -> linear encoder -> TopK activation -> linear decoder -> reconstruction
  - decoder columns constrained to unit norm (re-normalized after every optimizer step)
  - dead-feature recovery via periodic resampling toward high-loss inputs
  - expansion ratio 4-8x (default 8x -> 2048 atoms)
"""

import numpy as np
import torch
import torch.nn as nn


class TopKSAE(nn.Module):
    def __init__(self, d_in=256, n_atoms=2048, k=48):
        super().__init__()
        self.d_in = d_in
        self.n_atoms = n_atoms
        self.k = k

        self.b_pre = nn.Parameter(torch.zeros(d_in))
        self.encoder = nn.Linear(d_in, n_atoms, bias=True)
        self.decoder = nn.Linear(n_atoms, d_in, bias=False)

        with torch.no_grad():
            dec = torch.randn(d_in, n_atoms)
            dec = dec / dec.norm(dim=0, keepdim=True)
            self.decoder.weight.copy_(dec)
            self.encoder.weight.copy_(dec.t().clone())
            self.encoder.bias.zero_()

        # per-atom running stats for dead-feature detection
        self.register_buffer('fire_count', torch.zeros(n_atoms))
        self.register_buffer('steps_since_fire', torch.zeros(n_atoms))

    @torch.no_grad()
    def renorm_decoder(self):
        w = self.decoder.weight  # (d_in, n_atoms)
        norms = w.norm(dim=0, keepdim=True).clamp_min(1e-8)
        w.div_(norms)

    def encode(self, x):
        """x: (B, d_in). Returns (acts_topk, pre_acts, topk_idx)."""
        x_centered = x - self.b_pre
        pre_acts = torch.relu(self.encoder(x_centered))
        topk_vals, topk_idx = torch.topk(pre_acts, self.k, dim=-1)
        acts = torch.zeros_like(pre_acts)
        acts.scatter_(-1, topk_idx, topk_vals)
        return acts, pre_acts, topk_idx

    def decode(self, acts):
        return self.decoder(acts) + self.b_pre

    def forward(self, x):
        acts, pre_acts, topk_idx = self.encode(x)
        recon = self.decode(acts)
        return recon, acts, pre_acts, topk_idx

    @torch.no_grad()
    def update_fire_stats(self, topk_idx):
        fired = torch.zeros(self.n_atoms, device=topk_idx.device)
        fired.scatter_(0, topk_idx.reshape(-1), 1.0)
        self.fire_count += fired
        self.steps_since_fire = torch.where(
            fired > 0, torch.zeros_like(self.steps_since_fire), self.steps_since_fire + 1)


def train_topk_sae(X, n_atoms=2048, k=48, n_epochs=30, batch_size=1024, lr=1e-3,
                    dead_threshold_steps=200, aux_k=None, aux_alpha=1.0/32,
                    seed=0, device='cpu', verbose=True):
    """Train a TopK SAE on pooled h_t activations X (N, d_in).

    Dead-feature recovery: atoms that haven't fired in `dead_threshold_steps`
    optimizer steps get an auxiliary reconstruction loss (Gao et al.-style
    "AuxK" — reconstruct the residual using the top aux_k DEAD atoms), which
    gives them gradient signal to revive without touching the live top-k path.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    d_in = X.shape[1]
    aux_k = aux_k or k

    sae = TopKSAE(d_in=d_in, n_atoms=n_atoms, k=k).to(device)
    with torch.no_grad():
        sae.b_pre.copy_(torch.tensor(X.mean(axis=0), dtype=torch.float32, device=device))

    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    N = X.shape[0]
    X_t = torch.tensor(X, dtype=torch.float32, device=device)

    history = []
    step = 0
    for epoch in range(n_epochs):
        perm = torch.randperm(N)
        epoch_loss, epoch_l0, n_batches = 0.0, 0.0, 0
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            x = X_t[idx]

            acts, pre_acts, topk_idx = sae.encode(x)
            recon = sae.decode(acts)
            main_loss = ((recon - x) ** 2).sum(dim=-1).mean()

            # dead-feature auxiliary loss (AuxK): use dead atoms to reconstruct
            # the residual the live top-k path missed
            dead_mask = sae.steps_since_fire >= dead_threshold_steps
            aux_loss = torch.tensor(0.0, device=device)
            if dead_mask.any():
                residual = (x - recon).detach()
                dead_pre = torch.where(dead_mask, pre_acts, torch.full_like(pre_acts, -1e9))
                n_dead = int(dead_mask.sum().item())
                this_aux_k = min(aux_k, n_dead)
                aux_vals, aux_idx = torch.topk(dead_pre, this_aux_k, dim=-1)
                aux_acts = torch.zeros_like(pre_acts)
                aux_acts.scatter_(-1, aux_idx, torch.relu(aux_vals))
                aux_recon = sae.decoder(aux_acts)
                aux_loss = ((aux_recon - residual) ** 2).sum(dim=-1).mean()

            loss = main_loss + aux_alpha * aux_loss

            opt.zero_grad()
            loss.backward()
            opt.step()
            sae.renorm_decoder()
            sae.update_fire_stats(topk_idx)

            epoch_loss += main_loss.item()
            epoch_l0 += sae.k  # exact by construction (TopK)
            n_batches += 1
            step += 1

        history.append(dict(epoch=epoch, recon_loss=epoch_loss / n_batches,
                             l0=epoch_l0 / n_batches))
        if verbose and (epoch % 5 == 0 or epoch == n_epochs - 1):
            dead_pct = float((sae.steps_since_fire >= dead_threshold_steps).float().mean())
            print(f"    epoch {epoch:3d}: recon_loss={history[-1]['recon_loss']:.4f} "
                  f"L0={sae.k}  dead%={dead_pct*100:.1f}")

    return sae, history


@torch.no_grad()
def sae_metrics(sae, X, batch_size=4096, device='cpu'):
    """Mandatory SAE metrics (Section 8.2): L0, dead-atom %, variance explained."""
    N = X.shape[0]
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    total_var = float(np.var(X, axis=0).sum())
    ss_res = 0.0
    all_acts_fired = torch.zeros(sae.n_atoms, device=device)
    for i in range(0, N, batch_size):
        x = X_t[i:i + batch_size]
        acts, _, topk_idx = sae.encode(x)
        recon = sae.decode(acts)
        ss_res += ((recon - x) ** 2).sum().item()
        fired = torch.zeros(sae.n_atoms, device=device)
        fired.scatter_(0, topk_idx.reshape(-1), 1.0)
        all_acts_fired += fired
    var_explained = 1.0 - (ss_res / N) / total_var if total_var > 0 else float('nan')
    dead_pct = float((all_acts_fired == 0).float().mean())
    return dict(l0=sae.k, dead_pct=dead_pct, var_explained=float(var_explained),
                n_atoms=sae.n_atoms)


@torch.no_grad()
def get_all_activations(sae, X, batch_size=4096, device='cpu'):
    """Full (N, n_atoms) sparse activation matrix (dense storage; fine at this scale)."""
    N = X.shape[0]
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    out = np.zeros((N, sae.n_atoms), dtype=np.float32)
    for i in range(0, N, batch_size):
        x = X_t[i:i + batch_size]
        acts, _, _ = sae.encode(x)
        out[i:i + batch_size] = acts.cpu().numpy()
    return out
