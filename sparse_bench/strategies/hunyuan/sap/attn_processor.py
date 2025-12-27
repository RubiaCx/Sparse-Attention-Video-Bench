from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention

from sparse_bench.strategies.wan.sap.dynamic_map import identify_dynamic_map
from sparse_bench.strategies.wan.sap.flashinfer_dynamic import dynamic_block_sparse_attention_flashinfer
from sparse_bench.strategies.wan.sap.kmeans import batch_kmeans_euclid
from sparse_bench.strategies.wan.sap.permute import apply_inverse_permutation, permute_tensor_by_labels


class HunyuanAttn_SAPAttn_Processor2_0:
    """
    In-repo SAP(SVG2)-style processor for HunyuanVideo (no external Sparse-VideoGen repo required).

    This mirrors Hunyuan's attention processor call signature (returns (hidden_states, encoder_hidden_states)).
    """

    # Set by strategy (absolute semantics for SAP path)
    first_layers_fp: int = 0  # absolute layer count
    first_times_fp: int = 1001  # absolute timestep threshold in [0..1000]
    num_q_centroids: int = 400
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
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        timestep: Optional[int] = None,
    ):
        # Match existing Hunyuan SVG processor behavior: for some blocks, concat encoder tokens into hidden stream
        if attn.add_q_proj is None and encoder_hidden_states is not None:
            hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        # 1) QKV projections (self-attn on merged stream)
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()

        # 2) QK normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # 3) Rotary embeddings on latent stream (same as SVG processor)
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            if attn.add_q_proj is None and encoder_hidden_states is not None:
                query = torch.cat(
                    [
                        apply_rotary_emb(query[:, :, : -encoder_hidden_states.shape[1]], image_rotary_emb),
                        query[:, :, -encoder_hidden_states.shape[1] :],
                    ],
                    dim=2,
                )
                key = torch.cat(
                    [
                        apply_rotary_emb(key[:, :, : -encoder_hidden_states.shape[1]], image_rotary_emb),
                        key[:, :, -encoder_hidden_states.shape[1] :],
                    ],
                    dim=2,
                )
            else:
                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)

        # 4) Encoder condition projections if present (joint attention blocks)
        if attn.add_q_proj is not None and encoder_hidden_states is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)

            encoder_query = encoder_query.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()
            encoder_key = encoder_key.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()
            encoder_value = encoder_value.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()

            if attn.norm_added_q is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if attn.norm_added_k is not None:
                encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([query, encoder_query], dim=2)
            key = torch.cat([key, encoder_key], dim=2)
            value = torch.cat([value, encoder_value], dim=2)

        # 5) Attention (SAP)
        attn_out = self.attention_core_logic(query, key, value, timestep)
        attn_out = attn_out.transpose(1, 2).flatten(2, 3).to(query.dtype)

        # 6) Output projection + split back
        if encoder_hidden_states is not None:
            attn_out, encoder_hidden_states = (
                attn_out[:, : -encoder_hidden_states.shape[1]],
                attn_out[:, -encoder_hidden_states.shape[1] :],
            )
            if getattr(attn, "to_out", None) is not None:
                attn_out = attn.to_out[0](attn_out)
                attn_out = attn.to_out[1](attn_out)
            if getattr(attn, "to_add_out", None) is not None:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return attn_out, encoder_hidden_states

    @torch.no_grad()
    def _kmeans(self, x: torch.Tensor, *, is_query: bool):
        # x: [B*H, S, D]
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

        # Full-attention warmup
        full_attention_flag = False
        if self.layer_idx < self.first_layers_fp:
            full_attention_flag = True
        t0 = timestep[0] if isinstance(timestep, (tuple, list)) else timestep
        if hasattr(t0, "item"):
            t0 = t0.item()
        if t0 is None:
            t0 = 0
        if t0 > self.first_times_fp:
            full_attention_flag = True

        if full_attention_flag:
            if self.zero_step_kmeans_init and not self.centroids_init:
                q_bh = query.reshape(b * h, s, d).contiguous()
                k_bh = key.reshape(b * h, s, d).contiguous()
                _ = self._kmeans(q_bh, is_query=True)
                _ = self._kmeans(k_bh, is_query=False)
                self.centroids_init = True
            return F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)

        # SAP path
        q_bh = query.reshape(b * h, s, d).contiguous()
        k_bh = key.reshape(b * h, s, d).contiguous()

        qlabels, qcentroids, qc_sizes = self._kmeans(q_bh, is_query=True)
        klabels, kcentroids, kc_sizes = self._kmeans(k_bh, is_query=False)
        if not self.centroids_init:
            self.centroids_init = True
            print(f"[HunyuanSAP] Centroids initialized at layer {self.layer_idx}. Init step: {self.kmeans_iter_init}")

        dyn_map = identify_dynamic_map(
            qcentroids, kcentroids, top_p=self.top_p_kmeans, min_kc_ratio=self.min_kc_ratio
        )  # [B*H, QC, KC]
        dyn_map_bh = dyn_map.reshape(b, h, dyn_map.shape[1], dyn_map.shape[2])

        q_perm, q_sorted_idx = permute_tensor_by_labels(query, qlabels.reshape(b, h, s), dim=2)
        k_perm, _ = permute_tensor_by_labels(key, klabels.reshape(b, h, s), dim=2)
        v_perm, _ = permute_tensor_by_labels(value, klabels.reshape(b, h, s), dim=2)

        qc_sz = qc_sizes.reshape(b, h, -1).to(dtype=torch.int64)
        kc_sz = kc_sizes.reshape(b, h, -1).to(dtype=torch.int64)

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


