from __future__ import annotations

import math
import torch


@torch.no_grad()
def identify_dynamic_map(
    q_centroids: torch.Tensor,
    k_centroids: torch.Tensor,
    *,
    top_p: float,
    min_kc_ratio: float,
) -> torch.Tensor:
    """
    Build a boolean block map [B, QC, KC] selecting key-clusters per query-cluster.

    q_centroids: [B, QC, D]
    k_centroids: [B, KC, D]
    """
    if q_centroids.ndim != 3 or k_centroids.ndim != 3:
        raise ValueError("centroids must be [B, K, D]")
    b, qc, d = q_centroids.shape
    b2, kc, d2 = k_centroids.shape
    if b2 != b or d2 != d:
        raise ValueError("q/k centroids must share batch and dim")

    top_p = float(top_p)
    top_p = min(max(top_p, 0.0), 1.0)
    min_keep = int(math.ceil(float(min_kc_ratio) * kc))
    min_keep = max(1, min(min_keep, kc))

    # Similarity [B, QC, KC]
    sim = torch.einsum("bqd,bkd->bqk", q_centroids, k_centroids) / math.sqrt(d)
    probs = torch.softmax(sim, dim=-1)

    # Sort keys per query by prob desc
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)

    # keep until reaching top_p
    if top_p >= 1.0:
        k_keep = torch.full((b, qc), kc, device=probs.device, dtype=torch.int64)
    else:
        k_keep = (cumsum < top_p).sum(dim=-1) + 1  # at least 1
        k_keep = torch.clamp(k_keep, min=min_keep, max=kc)

    # boolean map
    mask = torch.zeros((b, qc, kc), device=probs.device, dtype=torch.bool)
    ar = torch.arange(kc, device=probs.device).view(1, 1, kc)
    keep_pos = ar < k_keep.unsqueeze(-1)  # [B,QC,KC] in sorted space
    kept_idx = torch.where(keep_pos, sorted_idx, torch.zeros_like(sorted_idx))
    mask.scatter_(dim=-1, index=kept_idx, src=keep_pos)
    return mask


