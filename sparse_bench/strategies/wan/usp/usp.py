import torch.distributed as dist
import torch.nn as nn
from diffusers.models.transformers.transformer_wan import WanAttention
from sparse_bench.morpher import AttributeReplacement, MethodReplacement, ModuleTransformation

from sparse_bench.strategies.base_strategy import VideoGenStrategy
from sparse_bench.strategies.registry import STRATEGIES

from .replacement import WanUSPAttnProcessor2_0, transformer_forward


@STRATEGIES.register("WanModel")
class WanUsp(VideoGenStrategy):
    def __init__(self, pipe, ring_size, ulysses_size):
        super().__init__(pipe)
        assert dist.is_initialized(), "WANRing only supports distributed inference"
        assert ring_size is not None or ulysses_size is not None, "Ring size or Ulysses size must be provided"
        self.ring_size = ring_size
        self.ulysses_size = ulysses_size

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
        attn_modules = [
            (name, int(name.split(".")[1]))
            for name, module in transformer.named_modules()
            if isinstance(module, WanAttention) and "attn1" in name
        ]
        attn_replacement = {
            name: ModuleTransformation(
                attribute_replacement=[AttributeReplacement(name="processor", value=WanUSPAttnProcessor2_0())]
            )
            for name, layer_idx in attn_modules
        }

        strategy = {
            "": ModuleTransformation(
                method_replacement=[MethodReplacement(name="forward", target_method=transformer_forward)]
            ),
            **attn_replacement,
        }
        return strategy
