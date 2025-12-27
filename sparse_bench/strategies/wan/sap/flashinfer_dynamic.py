from __future__ import annotations

from typing import Optional

import torch


@torch.no_grad()
def dynamic_block_sparse_attention_flashinfer(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    block_mask_map: torch.Tensor,
    block_row_sz: torch.Tensor,
    block_col_sz: torch.Tensor,
) -> torch.Tensor:
    """
    Flashinfer VariableBlockSparseAttentionWrapper wrapper, with basic version-compat handling.

    q,k,v: [B, H, S, D]
    block_mask_map: [B, H, QC, KC] bool
    block_row_sz:   [B, H, QC] int
    block_col_sz:   [B, H, KC] int
    """
    try:
        import flashinfer  # type: ignore
    except Exception as e:
        raise RuntimeError(f"flashinfer is required for SAP dynamic sparse attention: {e}")

    b, h, s, d = q.shape
    qc = block_row_sz.shape[-1]
    kc = block_col_sz.shape[-1]
    if block_mask_map.shape != (b, h, qc, kc):
        raise ValueError(f"block_mask_map must be {(b,h,qc,kc)}, got {tuple(block_mask_map.shape)}")

    # flashinfer wrapper expects per-head batch flattened
    q2 = q.reshape(b * h, s, d).contiguous()
    k2 = k.reshape(b * h, s, d).contiguous()
    v2 = v.reshape(b * h, s, d).contiguous()
    mask2 = block_mask_map.reshape(b * h, qc, kc).contiguous()
    row2 = block_row_sz.reshape(b * h, qc).contiguous()
    col2 = block_col_sz.reshape(b * h, kc).contiguous()

    # buffers
    float_workspace_buffer = torch.empty(128 * 1024 * 1024, device=q.device)
    vector_sparse_indices_buffer = torch.empty(1024 * 1024 * 1024, device=q.device)
    wrapper = flashinfer.sparse.VariableBlockSparseAttentionWrapper(float_workspace_buffer, backend="auto")

    # reset buffers (API changes across flashinfer versions)
    try:
        reset_kwargs = dict(
            float_workspace_buffer=getattr(wrapper, "_float_workspace_buffer", float_workspace_buffer),
            int_workspace_buffer=getattr(wrapper, "_int_workspace_buffer", None),
            vector_sparse_indices_buffer=vector_sparse_indices_buffer,
        )
        if hasattr(wrapper, "_vector_sparse_indptr_buffer"):
            reset_kwargs["vector_sparse_indptr_buffer"] = wrapper._vector_sparse_indptr_buffer
        reset_kwargs = {k: v for k, v in reset_kwargs.items() if v is not None}
        wrapper.reset_workspace_buffer(**reset_kwargs)
    except TypeError:
        try:
            wrapper.reset_workspace_buffer(
                float_workspace_buffer=float_workspace_buffer,
                vector_sparse_indices_buffer=vector_sparse_indices_buffer,
            )
        except Exception:
            pass

    wrapper.plan(
        block_mask_map=mask2,
        block_row_sz=row2,
        block_col_sz=col2,
        num_qo_heads=b * h,
        num_kv_heads=b * h,
        head_dim=d,
        q_data_type=q2.dtype,
        kv_data_type=k2.dtype,
    )
    out = wrapper.run(q2, k2, v2).reshape(b, h, s, d)
    return out


