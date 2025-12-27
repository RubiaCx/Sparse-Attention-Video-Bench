import torch.nn as nn
from diffusers import DiffusionPipeline
from diffusers.models.transformers.transformer_wan import WanAttention

from sparse_bench.strategies.base_strategy import VideoGenStrategy
from sparse_bench.strategies.registry import STRATEGIES

from .processor import WanXAttnProcessor


@STRATEGIES.register("WanXAttn")
class WanXAttn(VideoGenStrategy):
    def __init__(
        self,
        pipe: DiffusionPipeline,
        stride: int = 16,
        block_size: int = 128,
        use_triton: bool = True,
    ):
        super().__init__(pipe)
        self.stride = stride
        self.block_size = block_size
        self.use_triton = use_triton

    def preprocess(self, model: nn.Module):
        print(f"Applying WanXAttn Strategy: stride={self.stride}, block_size={self.block_size}")
        for name, module in model.named_modules():
            if isinstance(module, WanAttention):
                # 替换 Processor
                module.set_processor(
                    WanXAttnProcessor(
                        stride=self.stride,
                        block_size=self.block_size,
                        use_triton=self.use_triton
                    )
                )
        return model

    def get_module_transformation(self):
        return {}


