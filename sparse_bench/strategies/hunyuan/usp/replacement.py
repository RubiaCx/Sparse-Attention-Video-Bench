from typing import Any, Dict, Optional, Union

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_hunyuan_video import HunyuanVideoAttnProcessor2_0
from diffusers.utils import logging
from xfuser.core.distributed import get_sequence_parallel_rank, get_sequence_parallel_world_size, get_sp_group
from xfuser.core.long_ctx_attention import xFuserLongContextAttention
from diffusers.models.embeddings import apply_rotary_emb

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

USE_PEFT_BACKEND = False


def transformer_forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        pooled_projections: torch.Tensor,
        guidance: torch.Tensor = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p, p_t = self.config.patch_size, self.config.patch_size_t
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p
        post_patch_width = width // p
        first_frame_num_tokens = 1 * post_patch_height * post_patch_width


        # ===== USP START =====
        sp_size = get_sequence_parallel_world_size()
        sp_rank = get_sequence_parallel_rank()
        # ===== USP END =====

        # 1. RoPE
        image_rotary_emb = self.rope(hidden_states)

        # 2. Conditional embeddings
        temb, token_replace_emb = self.time_text_embed(timestep, pooled_projections, guidance)

        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states, timestep, encoder_attention_mask)

        # ===== USP START =====
        hidden_states = torch.chunk(hidden_states, sp_size, dim=1)[sp_rank]
        freqs_cos, freqs_sin = image_rotary_emb
        def get_rotary_emb_chunk(freqs):
            freqs = torch.chunk(freqs, sp_size, dim=0)[sp_rank]
            return freqs
        freqs_cos = get_rotary_emb_chunk(freqs_cos)
        freqs_sin = get_rotary_emb_chunk(freqs_sin)
        image_rotary_emb = (freqs_cos, freqs_sin)
        # ===== USP END =====

        # 3. Attention mask preparation
        # latent_sequence_length = hidden_states.shape[1]
        # condition_sequence_length = encoder_hidden_states.shape[1]
        # sequence_length = latent_sequence_length + condition_sequence_length
        # attention_mask = torch.ones(
        #     batch_size, sequence_length, device=hidden_states.device, dtype=torch.bool
        # )  # [B, N]
        # effective_condition_sequence_length = encoder_attention_mask.sum(dim=1, dtype=torch.int)  # [B,]
        # effective_sequence_length = latent_sequence_length + effective_condition_sequence_length
        # indices = torch.arange(sequence_length, device=hidden_states.device).unsqueeze(0)  # [1, N]
        # mask_indices = indices >= effective_sequence_length.unsqueeze(1)  # [B, N]
        # attention_mask = attention_mask.masked_fill(mask_indices, False)
        # attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, N]

        # ===== USP START =====
        effective_condition_sequence_length = encoder_attention_mask.sum(dim=1, dtype=torch.int)[0]  # [B,]
        encoder_hidden_states = encoder_hidden_states[
            :, :effective_condition_sequence_length, :
        ]
        attention_mask = None
        # ===== USP END =====

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.transformer_blocks:
                hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    attention_mask,
                    image_rotary_emb,
                    token_replace_emb,
                    first_frame_num_tokens,
                )

            for block in self.single_transformer_blocks:
                hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    attention_mask,
                    image_rotary_emb,
                    token_replace_emb,
                    first_frame_num_tokens,
                )

        else:
            for block in self.transformer_blocks:
                hidden_states, encoder_hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    attention_mask,
                    image_rotary_emb,
                    token_replace_emb,
                    first_frame_num_tokens,
                )

            for block in self.single_transformer_blocks:
                hidden_states, encoder_hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    attention_mask,
                    image_rotary_emb,
                    token_replace_emb,
                    first_frame_num_tokens,
                )

        # 5. Output projection
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = get_sp_group().all_gather(hidden_states, dim=-2)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1, p_t, p, p
        )
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (hidden_states,)

        return Transformer2DModelOutput(sample=hidden_states)

class HunyuanUSPAttnProcessor2_0(HunyuanVideoAttnProcessor2_0):
    def __init__(self):
        super().__init__()
        self.use_long_ctx_attn_kvcache = True
        from xfuser.core.long_ctx_attention import (
            xFuserLongContextAttention,
        )
        self.usp = xFuserLongContextAttention(
            use_kv_cache=self.use_long_ctx_attn_kvcache
        )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        batch_size, _, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attn.add_q_proj is None and encoder_hidden_states is not None:
            hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        # 1. QKV projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        # 2. QK normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # 3. Rotational positional embeddings applied to latent stream
        if image_rotary_emb is not None:
            if attn.add_q_proj is None and encoder_hidden_states is not None:
                query = torch.cat(
                    [
                        apply_rotary_emb(
                            query[:, :, : -encoder_hidden_states.shape[1]],
                            image_rotary_emb,
                        ),
                        query[:, :, -encoder_hidden_states.shape[1] :],
                    ],
                    dim=2,
                )
                key = torch.cat(
                    [
                        apply_rotary_emb(
                            key[:, :, : -encoder_hidden_states.shape[1]],
                            image_rotary_emb,
                        ),
                        key[:, :, -encoder_hidden_states.shape[1] :],
                    ],
                    dim=2,
                )
            else:
                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)

        # 4. Encoder condition QKV projection and normalization
        if attn.add_q_proj is not None and encoder_hidden_states is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)

            encoder_query = encoder_query.unflatten(2, (attn.heads, -1)).transpose(
                1, 2
            )
            encoder_key = encoder_key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            encoder_value = encoder_value.unflatten(2, (attn.heads, -1)).transpose(
                1, 2
            )

            if attn.norm_added_q is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if attn.norm_added_k is not None:
                encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([query, encoder_query], dim=2)
            key = torch.cat([key, encoder_key], dim=2)
            value = torch.cat([value, encoder_value], dim=2)

        if encoder_hidden_states is not None:
            num_encoder_hidden_states_tokens = encoder_hidden_states.shape[1]
            num_query_tokens = query.shape[2] - num_encoder_hidden_states_tokens
        else:
            # num_encoder_hidden_states_tokens = (
            #     get_runtime_state().max_condition_sequence_length
            # )
            num_encoder_hidden_states_tokens = 256
            num_query_tokens = query.shape[2] - num_encoder_hidden_states_tokens

        #! ---------------------------------------- ATTENTION ----------------------------------------
        query, encoder_query = query.split(
            [num_query_tokens, num_encoder_hidden_states_tokens], dim=2
        )
        key, encoder_key = key.split(
            [num_query_tokens, num_encoder_hidden_states_tokens], dim=2
        )
        value, encoder_value = value.split(
            [num_query_tokens, num_encoder_hidden_states_tokens], dim=2
        )

        encoder_query = encoder_query.transpose(1, 2)
        encoder_key = encoder_key.transpose(1, 2)
        encoder_value = encoder_value.transpose(1, 2)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        hidden_states = self.usp(
            None,
            query,
            key,
            value,
            dropout_p=0.0,
            causal=False,
            joint_tensor_query=encoder_query,
            joint_tensor_key=encoder_key,
            joint_tensor_value=encoder_value,
            joint_strategy="rear",
        )

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        # 6. Output projection
        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : -encoder_hidden_states.shape[1]],
                hidden_states[:, -encoder_hidden_states.shape[1] :],
            )

            if getattr(attn, "to_out", None) is not None:
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)

            if getattr(attn, "to_add_out", None) is not None:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states