"""
Task B — constrained-training projection hook.

Applies h_tilde_t = h_t - v_t (v_t^T h_t) immediately after the GRU output,
inside RSSM.observe_step and RSSM.imagine_step, so the projected h_tilde_t
(not the raw h_t) is what reaches all four consumers: the next GRU step
(via the returned h becoming next call's h), the prior head, the posterior
head, and the decoder (reward heads do not exist in this project's
WorldModel -- verified: no `reward` in src/model/world_model.py or rssm.py).

Because RSSM.observe_step and RSSM.imagine_step both compute h identically
(`h_next = self.gru(inp, h)`, established in Task 10 -- verified again here
directly from src/model/rssm.py), ONE hook at the GRU call covers both paths.

The base RSSM class (src/model/rssm.py) is NOT modified. This module wraps
an RSSM instance's `gru` submodule in place (monkeypatch on the instance,
not the class), so the exact same RSSM/WorldModel code used everywhere else
in this project is reused unmodified for conditions A and C; condition B
gets the identical code with the hook installed.

v_t is treated as a constant (stop-gradient) -- the projection is linear and
differentiable in h, but v_t itself never receives gradient. This is done by
wrapping v_t in torch.no_grad()-derived tensors before the projection.
"""

import torch
import torch.nn as nn


class ProjectionHookedGRU(nn.Module):
    """Wraps an existing nn.GRUCell. Forward computes the normal GRU output,
    then projects it: h' = h - v (v^T h), using the currently-set direction
    `self.v` (a (deter,) tensor, stop-gradient, updated externally by the
    training loop's refit schedule). If `self.v` is None, this is a no-op
    passthrough (used for condition A, the unconstrained control, so the
    exact same code path can be reused with the hook simply disabled)."""

    def __init__(self, base_gru):
        super().__init__()
        self.base_gru = base_gru
        self.v = None  # (deter,) tensor or None; set externally, never given grad

    def forward(self, inp, h):
        h_next = self.base_gru(inp, h)
        if self.v is None:
            return h_next
        v = self.v.detach().to(h_next.device, h_next.dtype)
        proj = (h_next * v).sum(dim=-1, keepdim=True) * v.unsqueeze(0)
        return h_next - proj

    def set_direction(self, v):
        """v: 1-D tensor/array, will be normalized to unit length and detached."""
        if v is None:
            self.v = None
            return
        v = torch.as_tensor(v, dtype=torch.float32)
        v = v / (v.norm() + 1e-12)
        self.v = v.detach()


def install_projection_hook(rssm):
    """Wraps rssm.gru in place with a ProjectionHookedGRU. Returns the hook
    object (call hook.set_direction(v) to update/clear the constraint).
    Idempotent: if rssm.gru is already a ProjectionHookedGRU, returns it
    unchanged rather than double-wrapping."""
    if isinstance(rssm.gru, ProjectionHookedGRU):
        return rssm.gru
    hook = ProjectionHookedGRU(rssm.gru)
    rssm.gru = hook
    return hook


def constraint_effectiveness(hook, h_batch):
    """Diagnostic for the pilot (§3.6.1): |v^T h_tilde| should be ~0 after
    projection. h_batch: (N, deter) tensor of POST-projection h values
    (i.e., collected downstream of the hook, as ordinarily produced during
    training/inference). Returns the mean absolute residual projection."""
    if hook.v is None:
        return float('nan')
    v = hook.v
    with torch.no_grad():
        resid = (h_batch.to(v.dtype) @ v).abs().mean().item()
    return resid
