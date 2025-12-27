from __future__ import annotations

from typing import Tuple

import torch


def permute_tensor_by_labels(
    x: torch.Tensor, labels: torch.Tensor, *, dim: int = 2
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Permute `x` along `dim` by sorting `labels`.

    - x: [B, H, S, D] (or compatible), permute along S (dim=2)
    - labels: [B*H, S] or [B, H, S]
    Returns:
      - x_perm: permuted tensor
      - sorted_indices: indices used for permutation (same leading dims as labels)
    """
    if dim != 2:
        raise ValueError("Only dim=2 is supported for now.")

    if labels.ndim == 3:
        b, h, s = labels.shape
        labels_bh = labels.reshape(b * h, s)
    elif labels.ndim == 2:
        labels_bh = labels
        s = labels.shape[-1]
    else:
        raise ValueError(f"labels must be 2D or 3D, got {labels.ndim}D")

    b, h, s_x, d = x.shape
    if s_x != s:
        raise ValueError(f"x seq_len={s_x} but labels seq_len={s}")

    # [B*H, S]
    sorted_indices = torch.argsort(labels_bh, dim=-1)

    # gather along dim=2
    x_bh = x.reshape(b * h, s, d)
    idx = sorted_indices.unsqueeze(-1).expand(-1, -1, d)
    x_perm = torch.gather(x_bh, dim=1, index=idx).reshape(b, h, s, d)

    return x_perm, sorted_indices


def apply_inverse_permutation(
    x_permuted: torch.Tensor, sorted_indices: torch.Tensor, *, dim: int = 2
) -> torch.Tensor:
    """
    Inverse of permute_tensor_by_labels for dim=2.

    - x_permuted: [B, H, S, D]
    - sorted_indices: [B*H, S] indices used in forward permutation
    Returns x in original order [B, H, S, D]
    """
    if dim != 2:
        raise ValueError("Only dim=2 is supported for now.")

    b, h, s, d = x_permuted.shape
    if sorted_indices.ndim != 2 or sorted_indices.shape[0] != b * h or sorted_indices.shape[1] != s:
        raise ValueError(
            f"sorted_indices must be [B*H,S]=[{b*h},{s}], got {tuple(sorted_indices.shape)}"
        )

    x_bh = x_permuted.reshape(b * h, s, d)
    out = torch.empty_like(x_bh)
    idx = sorted_indices.unsqueeze(-1).expand(-1, -1, d)
    out.scatter_(dim=1, index=idx, src=x_bh)
    return out.reshape(b, h, s, d)


