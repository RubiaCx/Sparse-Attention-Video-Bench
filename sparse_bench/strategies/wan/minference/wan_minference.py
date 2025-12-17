import torch.nn as nn
from diffusers import DiffusionPipeline
from diffusers.models.transformers.transformer_wan import WanAttention

from sparse_bench.strategies.base_strategy import VideoGenStrategy
from sparse_bench.strategies.registry import STRATEGIES
from .replacement import WanMInferenceAttnProcessor


@STRATEGIES.register("WanModel")
class WanMInference(VideoGenStrategy):
    def __init__(
        self,
        pipe: DiffusionPipeline,
    ):
        super().__init__(pipe)

    def preprocess(self, model: nn.Module):
        for name, module in model.named_modules():
            if isinstance(module, WanAttention) and "attn1" in name:
                module.set_processor(WanMInferenceAttnProcessor())
        return model

    def get_module_transformation(self):
        return {}
