import torch.distributed as dist
import torch.nn as nn
from diffusers.models.attention_processor import Attention
from sparse_bench.morpher import AttributeReplacement, MethodReplacement, ModuleTransformation

from sparse_bench.strategies.base_strategy import VideoGenStrategy
from sparse_bench.strategies.registry import STRATEGIES

from .replacement import transformer_forward, HunyuanVideoSpargeAttnProcessor2_0


@STRATEGIES.register("HunyuanModel")
class HunyuanSparge(VideoGenStrategy):

    def preprocess(self, model: nn.Module):
        for block in model.transformer_blocks:
            block.attn.processor = HunyuanVideoSpargeAttnProcessor2_0()
        
        for block in model.single_transformer_blocks:
            block.attn.processor = HunyuanVideoSpargeAttnProcessor2_0()
        
        return model

    def get_module_transformation(self):
        strategy = {
            "": ModuleTransformation(
                method_replacement=[MethodReplacement(name="forward", target_method=transformer_forward)]
            ),
        }
        return strategy
