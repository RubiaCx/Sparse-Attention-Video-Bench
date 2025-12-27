from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch.nn as nn
from diffusers import DiffusionPipeline

from sparse_bench.strategies.base_strategy import VideoGenStrategy
from sparse_bench.strategies.registry import STRATEGIES
from sparse_bench.strategies.wan.svg.modules import replace_sparse_forward

from .attn_processor import WanAttn_SAPAttn_Processor


@dataclass(frozen=True)
class SAPConfig:
    height: int
    width: int
    num_frames: int
    first_layers_fp: int
    first_times_fp: int
    num_q_centroids: int
    num_k_centroids: int
    top_p_kmeans: float
    min_kc_ratio: float
    kmeans_iter_init: int
    kmeans_iter_step: int
    zero_step_kmeans_init: bool
    logging_file: Optional[str]


@STRATEGIES.register("WanModel")
class WanSAP(VideoGenStrategy):
    """
    SAP (aka SVG2) integration implemented in-repo (no external Sparse-VideoGen repo required).
    """

    def __init__(
        self,
        pipe: DiffusionPipeline,
        *,
        height: int,
        width: int,
        num_frames: int,
        first_layers_fp: int,
        first_times_fp: int,
        num_q_centroids: int,
        num_k_centroids: int,
        top_p_kmeans: float,
        min_kc_ratio: float,
        kmeans_iter_init: int,
        kmeans_iter_step: int,
        zero_step_kmeans_init: bool,
        logging_file: Optional[str] = None,
    ):
        super().__init__(pipe)
        self.cfg = SAPConfig(
            height=height,
            width=width,
            num_frames=num_frames,
            first_layers_fp=first_layers_fp,
            first_times_fp=first_times_fp,
            num_q_centroids=num_q_centroids,
            num_k_centroids=num_k_centroids,
            top_p_kmeans=top_p_kmeans,
            min_kc_ratio=min_kc_ratio,
            kmeans_iter_init=kmeans_iter_init,
            kmeans_iter_step=kmeans_iter_step,
            zero_step_kmeans_init=zero_step_kmeans_init,
            logging_file=logging_file,
        )

    def preprocess(self, model: nn.Module):
        # Ensure Wan transformer forward passes `timestep` into attention (same as SVG path).
        replace_sparse_forward()

        AttnModule = WanAttn_SAPAttn_Processor
        AttnModule.context_length = 0
        AttnModule.first_layers_fp = int(self.cfg.first_layers_fp)
        AttnModule.first_times_fp = int(self.cfg.first_times_fp)
        AttnModule.num_q_centroids = int(self.cfg.num_q_centroids)
        AttnModule.num_k_centroids = int(self.cfg.num_k_centroids)
        AttnModule.top_p_kmeans = float(self.cfg.top_p_kmeans)
        AttnModule.min_kc_ratio = float(self.cfg.min_kc_ratio)
        AttnModule.kmeans_iter_init = int(self.cfg.kmeans_iter_init)
        AttnModule.kmeans_iter_step = int(self.cfg.kmeans_iter_step)
        AttnModule.zero_step_kmeans_init = bool(self.cfg.zero_step_kmeans_init)
        AttnModule.logging_file = self.cfg.logging_file

        for layer_idx, m in enumerate(model.blocks):
            m.attn1.set_processor(AttnModule(layer_idx))
        return model

    def get_module_transformation(self):
        return {}


