import torch.distributed as dist
import torch.nn as nn
from diffusers.models.attention_processor import Attention
from sparse_bench.morpher import AttributeReplacement, MethodReplacement, ModuleTransformation

from sparse_bench.strategies.base_strategy import VideoGenStrategy
from sparse_bench.strategies.registry import STRATEGIES

from .replacement import transformer_forward


@STRATEGIES.register("HunyuanModel")
class HunyuanDense(VideoGenStrategy):

    def get_module_transformation(self):
        strategy = {
            "": ModuleTransformation(
                method_replacement=[MethodReplacement(name="forward", target_method=transformer_forward)]
            ),
        }
        return strategy
