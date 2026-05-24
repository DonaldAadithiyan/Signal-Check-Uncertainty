import os
import torch

def _auto_device():
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

XS_CONFIG = dict(
    # RSSM XS configuration (DreamerV3 paper Table 1)
    rssm_deter=256,
    rssm_hidden=256,
    rssm_stoch=32,
    rssm_classes=32,
    embed_dim=64,
    obs_dim=5,   # cartpole_swingup: position(3) + velocity(2)
    act_dim=1,   # cartpole slider force

    # Training
    seq_len=16,
    batch_size=8,
    total_env_steps=100_000,
    warmup_steps=1_000,
    replay_capacity=600,   # max episodes in buffer
    kl_free=1.0,
    kl_scale=1.0,
    lr=3e-4,
    grad_clip=1.0,
    train_every=1,         # gradient steps per env step

    # Collection
    n_eval_episodes=20,
    noise_std=0.1,         # OOD noise for Set B
    episode_max_steps=500,

    # Probe B
    rollout_samples=5,
    rollout_horizon=5,

    # Ensemble
    ensemble_seeds=[0, 1, 2],

    # Device (auto-detect MPS on Apple Silicon, else CPU)
    device=_auto_device(),

    # Paths
    output_dir='outputs',
    checkpoint_path='outputs/checkpoints/world_model.pt',
    training_data_path='outputs/data/training_states.npz',
    set_a_path='outputs/data/set_a_id.npz',
    set_b_path='outputs/data/set_b_ood.npz',
    set_c_path='outputs/data/set_c_contrastive.npz',
    probe_results_path='outputs/results/probe_results.csv',
    block_auroc_path='outputs/results/block_auroc.csv',
    ht_vs_zt_path='outputs/results/ht_vs_zt.csv',
    figures_dir='outputs/figures',
)
