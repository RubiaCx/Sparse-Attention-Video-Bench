import torch
import torch.nn as nn
from diffusers import DiffusionPipeline
from diffusers.models.attention_processor import Attention

from sparse_bench.strategies.base_strategy import VideoGenStrategy
from sparse_bench.strategies.registry import STRATEGIES

from .attn_processor import WanAttn_SparseAttn_Processor2_0, prepare_flexattention
from .modules import replace_sparse_forward
from .utils import get_attention_mask, sparsity_to_width


@STRATEGIES.register("WanModel")
class WanSVG(VideoGenStrategy):
    def __init__(
        self,
        pipe: DiffusionPipeline,
        height: int,
        width: int,
        num_frames: int,
        num_sampled_rows: int,
        sample_mse_max_row: int,
        sparsity: float,
        first_layers_fp: int,
        first_times_fp: int,
    ):
        super().__init__(pipe)
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.num_sampled_rows = num_sampled_rows
        self.sample_mse_max_row = sample_mse_max_row
        self.sparsity = sparsity
        self.first_layers_fp = first_layers_fp
        self.first_times_fp = first_times_fp

    def preprocess(self, model: nn.Module):
        masks = ["spatial", "temporal"]

        context_length = 0
        num_frame = 1 + self.num_frames // (
            self.pipe.vae_scale_factor_temporal * self.pipe.transformer.config.patch_size[0]
        )
        mod_value = self.pipe.vae_scale_factor_spatial * self.pipe.transformer.config.patch_size[1]
        frame_size = int(self.height // mod_value) * int(self.width // mod_value)

        dtype = torch.bfloat16

        AttnModule = WanAttn_SparseAttn_Processor2_0
        AttnModule.num_sampled_rows = self.num_sampled_rows
        AttnModule.sample_mse_max_row = self.sample_mse_max_row
        AttnModule.attention_masks = [
            get_attention_mask(mask_name, self.sample_mse_max_row, context_length, num_frame, frame_size)
            for mask_name in masks
        ]
        AttnModule.first_layers_fp = self.first_layers_fp
        AttnModule.first_times_fp = self.first_times_fp

        multiplier = diag_width = sparsity_to_width(self.sparsity, context_length, num_frame, frame_size)

        AttnModule.context_length = context_length
        AttnModule.num_frame = num_frame
        AttnModule.frame_size = frame_size

        # NOTE: ??? Prepare placement will strongly decrease PSNR
        # prepare_placement(2, 48, 64, dtype, "cuda", context_length, num_frame, frame_size)
        block_mask = prepare_flexattention(
            1, 40, 128, dtype, "cuda", context_length, context_length, num_frame, frame_size, diag_width, multiplier
        )
        AttnModule.block_mask = block_mask

        print(block_mask)

        replace_sparse_forward()

        num_layers = len(model.blocks)

        for layer_idx, m in enumerate(model.blocks):
            m.attn1.processor.layer_idx = layer_idx

        for layer_idx, m in enumerate(model.blocks):
            m.attn1.set_processor(AttnModule(layer_idx))
            m.attn1.processor.num_layers = num_layers

    def get_module_transformation(self):
        return {}
