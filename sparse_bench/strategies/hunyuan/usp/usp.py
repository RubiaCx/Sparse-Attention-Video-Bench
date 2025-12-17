import torch.distributed as dist
import torch.nn as nn
from diffusers.models.attention_processor import Attention
from sparse_bench.morpher import AttributeReplacement, MethodReplacement, ModuleTransformation

from sparse_bench.strategies.base_strategy import VideoGenStrategy
from sparse_bench.strategies.registry import STRATEGIES

from .replacement import transformer_forward, HunyuanUSPAttnProcessor2_0


@STRATEGIES.register("HunyuanModel")
class HunyuanUsp(VideoGenStrategy):
    def __init__(self, pipe, ulysses_size, ring_size):
        super().__init__(pipe)
        assert ulysses_size is not None and ring_size is not None, "Ulysses size and ring size must be specified"
        self.ulysses_size = ulysses_size
        self.ring_size = ring_size
        assert dist.is_initialized(), "WANRing only supports distributed inference"

    def preprocess(self, model: nn.Module):
        import xfuser.core.distributed.parallel_state as parallel_state
        from xfuser.core.distributed import initialize_model_parallel
        from xfuser.core.distributed.parallel_state import init_world_group

        # set the _WORLD global variable in xfuser
        ranks = list(range(dist.get_world_size()))
        parallel_state._WORLD = init_world_group(ranks, dist.get_rank(), backend="nccl")
        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=self.ring_size,
            ulysses_degree=self.ulysses_size,
        )
        return model

    def get_module_transformation(self):
        transformer = self.pipe.transformer
        attn_module_names = [
            name
            for name, module in transformer.named_modules()
            if isinstance(module, Attention) and 'transformer_block' in name
        ]
        attn_replacement = {
            name: ModuleTransformation(
                attribute_replacement=[AttributeReplacement(name="processor", value=HunyuanUSPAttnProcessor2_0())]
            )
            for name in attn_module_names
        }

        strategy = {
            "": ModuleTransformation(
                method_replacement=[MethodReplacement(name="forward", target_method=transformer_forward)]
            ),
            **attn_replacement,
        }
        return strategy
