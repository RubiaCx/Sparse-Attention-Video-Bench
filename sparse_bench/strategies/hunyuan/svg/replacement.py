import torch
from typing import Optional, Tuple, Dict, Any, Union
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
USE_PEFT_BACKEND = False


def tranformer_block_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    *args,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # 1. Input normalization
    norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
    norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
        encoder_hidden_states, emb=temb
    )

    timestep = kwargs.pop("timestep", None)

    # 2. Joint attention
    attn_output, context_attn_output = self.attn(
        hidden_states=norm_hidden_states,
        encoder_hidden_states=norm_encoder_hidden_states,
        attention_mask=attention_mask,
        image_rotary_emb=freqs_cis,
        timestep=timestep,
    )

    # 3. Modulation and residual connection
    hidden_states = hidden_states + attn_output * gate_msa.unsqueeze(1)
    encoder_hidden_states = encoder_hidden_states + context_attn_output * c_gate_msa.unsqueeze(1)

    norm_hidden_states = self.norm2(hidden_states)
    norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)

    norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
    norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

    # 4. Feed-forward
    ff_output = self.ff(norm_hidden_states)
    context_ff_output = self.ff_context(norm_encoder_hidden_states)

    hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output
    encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

    return hidden_states, encoder_hidden_states


def single_transformer_block_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    *args,
    **kwargs,
) -> torch.Tensor:
    text_seq_length = encoder_hidden_states.shape[1]
    hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)
    timestep = kwargs.pop("timestep", None)

    residual = hidden_states

    # 1. Input normalization
    norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
    mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

    norm_hidden_states, norm_encoder_hidden_states = (
        norm_hidden_states[:, :-text_seq_length, :],
        norm_hidden_states[:, -text_seq_length:, :],
    )

    # 2. Attention
    attn_output, context_attn_output = self.attn(
        hidden_states=norm_hidden_states,
        encoder_hidden_states=norm_encoder_hidden_states,
        attention_mask=attention_mask,
        image_rotary_emb=image_rotary_emb,
        timestep=timestep,
    )
    attn_output = torch.cat([attn_output, context_attn_output], dim=1)

    # 3. Modulation and residual connection
    hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
    hidden_states = gate.unsqueeze(1) * self.proj_out(hidden_states)
    hidden_states = hidden_states + residual

    hidden_states, encoder_hidden_states = (
        hidden_states[:, :-text_seq_length, :],
        hidden_states[:, -text_seq_length:, :],
    )
    return hidden_states, encoder_hidden_states




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

        # 1. RoPE
        image_rotary_emb = self.rope(hidden_states)

        # 2. Conditional embeddings
        temb, token_replace_emb = self.time_text_embed(timestep, pooled_projections, guidance)

        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states, timestep, encoder_attention_mask)

        # 3. Attention mask preparation
        latent_sequence_length = hidden_states.shape[1]
        condition_sequence_length = encoder_hidden_states.shape[1]
        sequence_length = latent_sequence_length + condition_sequence_length
        attention_mask = torch.ones(
            batch_size, sequence_length, device=hidden_states.device, dtype=torch.bool
        )  # [B, N]
        effective_condition_sequence_length = encoder_attention_mask.sum(dim=1, dtype=torch.int)  # [B,]
        effective_sequence_length = latent_sequence_length + effective_condition_sequence_length
        indices = torch.arange(sequence_length, device=hidden_states.device).unsqueeze(0)  # [1, N]
        mask_indices = indices >= effective_sequence_length.unsqueeze(1)  # [B, N]
        attention_mask = attention_mask.masked_fill(mask_indices, False)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, N]
        # attention_mask = None
        encoder_hidden_states = encoder_hidden_states[:, :effective_condition_sequence_length, :]
        attention_mask = None

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
                    timestep=timestep
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
                    timestep=timestep
                )

        # 5. Output projection
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

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