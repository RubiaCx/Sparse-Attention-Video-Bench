import torch
import torch.nn.functional as F
from diffusers.models.transformers.transformer_wan import WanAttention

try:
    # 尝试导入 XAttention，根据 compare.py 的路径
    # 假设 xattn 包已经安装或在 PYTHONPATH 中
    from xattn.src.Xattention import Xattention_prefill
except ImportError:
    # 如果没安装，提供一个假的实现或者报错，防止 import 崩溃
    # 但实际运行时会报错
    Xattention_prefill = None

class WanXAttnProcessor:
    def __init__(self, stride=16, block_size=128, use_triton=True, chunk_size=None):
        self.stride = stride
        self.block_size = block_size
        self.use_triton = use_triton
        self.chunk_size = chunk_size

    def __call__(
        self,
        attn: WanAttention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        image_rotary_emb: torch.Tensor = None,
    ) -> torch.Tensor:
        
        # 1. QKV Projection
        # WanAttention 的实现通常是 self-attention
        # hidden_states: [batch, seq_len, dim]
        
        # WanAttention 源码逻辑复现 (简化版):
        # query = attn.to_q(hidden_states)
        # key = attn.to_k(hidden_states)
        # value = attn.to_v(hidden_states)
        
        # 但是 Diffusers 的 WanAttention 可能已经把 QKV 融合了，或者分开了
        # 我们需要适配它的 forward 签名。
        # 参考 standard WanAttention forward:
        
        batch_size, sequence_length, _ = hidden_states.shape

        # Projection
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # Multi-head reshape
        # [batch, seq_len, heads, head_dim] -> [batch, heads, seq_len, head_dim]
        query = query.view(batch_size, sequence_length, attn.heads, attn.head_dim).transpose(1, 2)
        key = key.view(batch_size, sequence_length, attn.heads, attn.head_dim).transpose(1, 2)
        value = value.view(batch_size, sequence_length, attn.heads, attn.head_dim).transpose(1, 2)

        # Apply Rotary Embeddings (RoPE) if provided
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            # Wan 可能有特殊的 RoPE应用方式，这里假设是标准的
            # 注意：WanAttention 可能把 RoPE 放在了 processor 外面或者里面
            # 如果是 Processor 负责，我们需要应用它
            # 但 xattn 可能不支持 RoPE？或者我们需要在传给 xattn 之前应用 RoPE
            
            # 暂时假设 RoPE 在这里应用
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # Transpose back to [batch, heads, seq_len, dim] for XAttention
        # Wait, compare.py says: 
        #   q, k, v shape: [bsz, heads, seq_len, dim]
        # So we are good with current shape.

        # Call XAttention
        if Xattention_prefill is None:
            raise ImportError("Could not import Xattention_prefill. Please check xattn installation.")

        # XAttention expects [batch, heads, seq_len, head_dim]
        # It handles the sparse calculation internally
        
        # 注意: causal=True 在 Video 生成中通常不一定适用（除非是纯自回归模型）
        # Wan 是 Diffusion Transformer，通常是 Bidirectional 的 (causal=False)
        # 除非是 Temporal Attention 且是 Causal 的。
        # 但 Wan 是 T2V，通常全部 token 可见。这里我们暂且设为 False，或者留给参数控制。
        # compare.py 里设为了 True，可能是为了测试。对于 DiT，通常是 False。
        is_causal = False 

        hidden_states = Xattention_prefill(
            query_states=query,
            key_states=key,
            value_states=value,
            stride=self.stride,
            block_size=self.block_size,
            use_triton=self.use_triton,
            causal=is_causal,
            chunk_size=self.chunk_size
        )

        # Output: [batch, heads, seq_len, head_dim] -> [batch, seq_len, heads, head_dim]
        hidden_states = hidden_states.transpose(1, 2).contiguous()
        hidden_states = hidden_states.reshape(batch_size, sequence_length, -1)

        # Output Projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


