from __future__ import annotations

from typing import Optional, Tuple

import torch


@torch.no_grad()
def batch_kmeans_euclid(
    x: torch.Tensor,
    *,
    n_clusters: int,
    max_iters: int,
    init_centroids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Simple batched kmeans (euclidean) in torch.

    x: [B, N, D]
    Returns:
      labels: [B, N] (int64)
      centroids: [B, K, D]
      cluster_sizes: [B, K] (int64)
      n_iters: int
    """
    if x.ndim != 3:
        raise ValueError(f"x must be [B,N,D], got {tuple(x.shape)}")
    b, n, d = x.shape
    k = int(n_clusters)
    if k <= 0:
        raise ValueError("n_clusters must be > 0")

    if init_centroids is None:
        # random init from points
        idx = torch.randint(0, n, (b, k), device=x.device)
        centroids = torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, d)).contiguous()
    else:
        centroids = init_centroids.to(device=x.device, dtype=x.dtype).contiguous()
        if centroids.shape != (b, k, d):
            raise ValueError(f"init_centroids must be {(b,k,d)}, got {tuple(centroids.shape)}")

    # Ensure we execute at least one assignment (for max_iters <= 0)
    iters = max(1, int(max_iters))
    labels = torch.empty((b, n), device=x.device, dtype=torch.int64)
    for it in range(iters):
        # distances: [B, N, K]
        # (x - c)^2 = x^2 + c^2 - 2 x·c
        x2 = (x * x).sum(-1, keepdim=True)  # [B,N,1]
        c2 = (centroids * centroids).sum(-1).unsqueeze(1)  # [B,1,K]
        xc = torch.einsum("bnd,bkd->bnk", x, centroids)  # [B,N,K]
        dist = x2 + c2 - 2 * xc

        labels = dist.argmin(dim=-1)  # [B,N]

        # update centroids
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros((b, k), device=x.device, dtype=torch.int64)
        counts.scatter_add_(1, labels, torch.ones_like(labels, dtype=torch.int64))

        # sum points per cluster
        new_centroids.scatter_add_(
            1, labels.unsqueeze(-1).expand(-1, -1, d), x
        )

        # avoid div by zero
        denom = counts.clamp_min(1).to(dtype=x.dtype).unsqueeze(-1)
        centroids = (new_centroids / denom).contiguous()

    return labels, centroids, counts, iters


