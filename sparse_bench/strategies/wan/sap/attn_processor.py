from __future__ import annotations

import math
from typing import Any, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention

from .dynamic_map import identify_dynamic_map
from .flashinfer_dynamic import dynamic_block_sparse_attention_flashinfer
from .kmeans import batch_kmeans_euclid
from .permute import apply_inverse_permutation, permute_tensor_by_labels


class WanAttn_SAPAttn_Processor:
    """
    In-repo SAP(SVG2)-style processor (no external Sparse-VideoGen repo required).

    Notes:
    - This implementation keeps the same high-level pipeline: kmeans -> permutation -> dynamic block sparse attention.
    - It relies on `flashinfer` (pip dependency) for the variable-block sparse attention kernel.
    """

    # Set by strategy (class-level knobs, matching the existing style in SVG processors)
    context_length: int = 0
    first_layers_fp: int = 0  # absolute layer count
    first_times_fp: int = 1001  # absolute timestep threshold in [0..1000]
    num_q_centroids: int = 300
    num_k_centroids: int = 1000
    top_p_kmeans: float = 0.9
    min_kc_ratio: float = 0.10
    kmeans_iter_init: int = 50
    kmeans_iter_step: int = 2
    zero_step_kmeans_init: bool = False
    logging_file: Optional[str] = None

    def __init__(self, layer_idx: int):
        self.layer_idx = int(layer_idx)
        self.centroids_init = False
        self.q_centroids: Optional[torch.Tensor] = None
        self.k_centroids: Optional[torch.Tensor] = None

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Any] = None,
        timestep: Optional[int] = None,
    ) -> torch.Tensor:
        # Mirror the existing Wan SVG processor behavior for T2V/I2V.
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None and encoder_hidden_states is not None:
            encoder_hidden_states_img = encoder_hidden_states[:, :257]
            encoder_hidden_states = encoder_hidden_states[:, 257:]

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        # QKV
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()

        # Apply rotary.
        #
        # WAN's `rotary_emb` format varies across diffusers versions and in-repo forward
        # replacements. Other WAN strategies in this repo consistently normalize it into
        # (freqs_cos, freqs_sin) and apply RoPE explicitly; we follow that convention here.
        if rotary_emb is not None:
            def _normalize_rotary_emb(x: Any) -> Tuple[torch.Tensor, torch.Tensor]:
                # Common case: already (cos, sin)
                if isinstance(x, (tuple, list)):
                    if len(x) == 2:
                        return x[0], x[1]
                    if len(x) == 1:
                        return _normalize_rotary_emb(x[0])
                    raise ValueError(f"Unexpected rotary_emb sequence length: {len(x)}")

                # Some implementations provide an object with `cos`/`sin` tensors
                if hasattr(x, "cos") and hasattr(x, "sin") and not torch.is_tensor(x):
                    cos, sin = getattr(x, "cos"), getattr(x, "sin")
                    if torch.is_tensor(cos) and torch.is_tensor(sin):
                        return cos, sin

                # Tensor encodings
                if torch.is_tensor(x):
                    # Complex cis: real/imag
                    if x.is_complex():
                        return x.real, x.imag
                    # Stacked (2, ...)
                    if x.dim() >= 1 and x.shape[0] == 2:
                        return x[0], x[1]
                    # Packed (..., 2)
                    if x.dim() >= 1 and x.shape[-1] == 2:
                        return x[..., 0], x[..., 1]
                    # Fallback: treat as angles
                    return x.cos(), x.sin()

                # Iterable/generator
                if isinstance(x, Iterable):
                    return _normalize_rotary_emb(list(x))

                raise TypeError(f"Unsupported rotary_emb type: {type(x)}")

            def _apply_rotary_emb(
                hs: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ) -> torch.Tensor:
                # hs: [B, S, H, D]
                x1, x2 = hs.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hs)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hs)

            freqs_cos, freqs_sin = _normalize_rotary_emb(rotary_emb)
            # query/key are [B, H, S, D] -> RoPE helper expects [B, S, H, D]
            query = _apply_rotary_emb(query.transpose(1, 2), freqs_cos, freqs_sin).transpose(1, 2)
            key = _apply_rotary_emb(key.transpose(1, 2), freqs_cos, freqs_sin).transpose(1, 2)

        # I2V image branch (dense SDPA)
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)
            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            hidden_states_img = F.scaled_dot_product_attention(
                query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3).type_as(query)

        # Cross-attn in Wan uses timestep=None; keep dense
        if timestep is None:
            out = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
        else:
            out = self.attention_core_logic(query, key, value, timestep)

        out = out.transpose(1, 2).flatten(2, 3).type_as(query)
        if hidden_states_img is not None:
            out = out + hidden_states_img
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out

    @torch.no_grad()
    def _kmeans(self, x: torch.Tensor, *, is_query: bool) -> torch.Tensor:
        """
        x: [B*H, S, D]
        returns labels [B*H, S], updates centroids cache
        """
        b, s, d = x.shape
        n_clusters = self.num_q_centroids if is_query else self.num_k_centroids
        if not self.centroids_init:
            iters = self.kmeans_iter_init
            init = None
        else:
            iters = self.kmeans_iter_step
            init = self.q_centroids if is_query else self.k_centroids

        labels, centroids, cluster_sizes, _n_iters = batch_kmeans_euclid(
            x, n_clusters=n_clusters, max_iters=iters, init_centroids=init
        )
        if is_query:
            self.q_centroids = centroids
        else:
            self.k_centroids = centroids
        return labels, centroids, cluster_sizes

    @torch.no_grad()
    def attention_core_logic(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, timestep):
        # query/key/value: [B, H, S, D]
        b, h, s, d = query.shape

        # full-attention warmup
        full_attention_flag = False
        if self.layer_idx < self.first_layers_fp:
            full_attention_flag = True
        # timestep may be tensor-like
        t0 = timestep[0] if isinstance(timestep, (tuple, list)) else timestep
        if hasattr(t0, "item"):
            t0 = t0.item()
        if t0 > self.first_times_fp:
            full_attention_flag = True

        if full_attention_flag:
            if self.zero_step_kmeans_init and not self.centroids_init:
                # initialize centroids once using video tokens only (Wan T2V has context_length=0)
                q_bh = query.reshape(b * h, s, d).contiguous()
                k_bh = key.reshape(b * h, s, d).contiguous()
                _ = self._kmeans(q_bh, is_query=True)
                _ = self._kmeans(k_bh, is_query=False)
                self.centroids_init = True
            return F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)

        # --- SAP path ---
        q_bh = query.reshape(b * h, s, d).contiguous()
        k_bh = key.reshape(b * h, s, d).contiguous()
        v_bh = value.reshape(b * h, s, d).contiguous()

        qlabels, qcentroids, qc_sizes = self._kmeans(q_bh, is_query=True)
        klabels, kcentroids, kc_sizes = self._kmeans(k_bh, is_query=False)
        if not self.centroids_init:
            self.centroids_init = True
            print(f"Centroids initialized at layer {self.layer_idx}. Init step: {self.kmeans_iter_init}")

        # dynamic map based on centroid similarity
        dyn_map = identify_dynamic_map(
            qcentroids,
            kcentroids,
            top_p=self.top_p_kmeans,
            min_kc_ratio=self.min_kc_ratio,
        )

        # permute Q/K/V
        q_perm, q_sorted_idx = permute_tensor_by_labels(query, qlabels.reshape(b, h, s), dim=2)
        k_perm, k_sorted_idx = permute_tensor_by_labels(key, klabels.reshape(b, h, s), dim=2)
        v_perm, _ = permute_tensor_by_labels(value, klabels.reshape(b, h, s), dim=2)

        # row/col sizes & map in [B,H,QC/KC] on CPU for flashinfer planner
        qc_sz = qc_sizes.reshape(b, h, -1).to(dtype=torch.int64)
        kc_sz = kc_sizes.reshape(b, h, -1).to(dtype=torch.int64)
        dyn_map_bh = dyn_map.reshape(b, h, dyn_map.shape[1], dyn_map.shape[2])

        out_perm = dynamic_block_sparse_attention_flashinfer(
            q_perm,
            k_perm,
            v_perm,
            block_mask_map=dyn_map_bh.to(device=query.device),
            block_row_sz=qc_sz.to(device=query.device),
            block_col_sz=kc_sz.to(device=query.device),
        )

        out = apply_inverse_permutation(out_perm, q_sorted_idx, dim=2)
        return out


