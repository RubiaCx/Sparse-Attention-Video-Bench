import torch
import torch.nn as nn
from diffusers import DiffusionPipeline
from diffusers.models.attention_processor import Attention

from sparse_bench.strategies.base_strategy import VideoGenStrategy

from .attn_processor import HunyuanAttn_SparseAttn_Processor2_0, prepare_flexattention
from .modules import replace_sparse_forward
from .utils import get_attention_mask, sparsity_to_width


class HunyuanSVG(VideoGenStrategy):
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

        self.context_length = 256
        self.num_frame = 1 + self.num_frames // (
            self.pipe.vae_scale_factor_temporal
        )
        self.mod_value = self.pipe.vae_scale_factor_spatial * self.pipe.transformer.config.patch_size
        self.frame_size = int(self.height // self.mod_value) * int(self.width // self.mod_value)

        dtype = torch.bfloat16

        AttnModule = HunyuanAttn_SparseAttn_Processor2_0
        AttnModule.num_sampled_rows = self.num_sampled_rows
        AttnModule.sample_mse_max_row = self.sample_mse_max_row
        AttnModule.first_layers_fp = self.first_layers_fp
        AttnModule.first_times_fp = self.first_times_fp
        AttnModule.num_frame = self.num_frame
        AttnModule.frame_size = self.frame_size

        num_blocks = len(model.transformer_blocks) + len(model.single_transformer_blocks)

        for idx, block in enumerate(model.transformer_blocks + model.single_transformer_blocks):
            block.attn.set_processor(AttnModule(idx))
            block.attn.processor.num_layers = num_blocks
        return model

    def build_attn_masks(self, model, prompt_length):
        masks = ["spatial", "temporal"]

        attention_masks = [
            get_attention_mask(
                mask_name, 
                prompt_length, 
                self.num_frame, 
                self.frame_size,
                self.sample_mse_max_row
            )
            for mask_name in masks
        ]
        
        multiplier = diag_width = sparsity_to_width(self.sparsity, self.context_length, self.num_frame, self.frame_size)

        # NOTE: ??? Prepare placement will strongly decrease PSNR
        # prepare_placement(2, 48, 64, dtype, "cuda", context_length, num_frame, frame_size)
        block_mask = prepare_flexattention(
            1, 24, 128, torch.bfloat16, "cuda", prompt_length, prompt_length, self.num_frame, self.frame_size, diag_width, multiplier
        )

        for idx, block in enumerate(model.transformer_blocks + model.single_transformer_blocks):
            block.attn.processor.attention_masks = attention_masks
            block.attn.processor.block_mask = block_mask
            block.attn.processor.context_length = prompt_length
        

    def get_module_transformation(self):
        from diffusers.models.transformers.transformer_hunyuan_video import HunyuanVideoTransformerBlock, HunyuanVideoSingleTransformerBlock
        from .replacement import tranformer_block_forward, single_transformer_block_forward, transformer_forward
        from sparse_bench.morpher import MethodReplacement, ModuleTransformation
        return {
            "": ModuleTransformation(
                method_replacement=[MethodReplacement(name="forward", target_method=transformer_forward)]
            ),
            HunyuanVideoTransformerBlock: ModuleTransformation(
                method_replacement=[MethodReplacement(name="forward", target_method=tranformer_block_forward)]
            ),
            HunyuanVideoSingleTransformerBlock: ModuleTransformation(
                method_replacement=[MethodReplacement(name="forward", target_method=single_transformer_block_forward)]
            ),
        }
