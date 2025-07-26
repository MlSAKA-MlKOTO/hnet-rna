from dataclasses import asdict

import torch


def get_seq_idx(cu_seqlens, device=None):
    seq_idx = torch.zeros(cu_seqlens[-1], dtype=torch.long, device=device)
    seq_idx[cu_seqlens[:-1]] = 1
    seq_idx = (torch.cumsum(seq_idx, dim=0) - 1).unsqueeze(0).int()

    return seq_idx


def get_stage_cfg(cfg, stage_idx):
    return {
        k: v[stage_idx] if isinstance(v, list) else v for k, v in asdict(cfg).items()
    }
