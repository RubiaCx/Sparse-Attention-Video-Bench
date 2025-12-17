# from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from sparse_bench.strategies.base_strategy import VideoGenStrategy
from diffusers.utils import export_to_video
from sparse_bench.morpher import Morpher
from sparse_bench.utils import get_timestamp
import re

from sparse_bench.strategies.hunyuan import HunyuanUsp, HunyuanSVG, HunyuanDense, HunyuanSparge
from sparse_bench.utils import init_distributed, is_distributed, is_rank_zero

_DEFAULT_PROMPT = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."
_DEFAULT_NEGATIVE_PROMPT = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"


def set_jit_enabled(enabled: bool):
    """Enables/disables JIT"""
    if torch.__version__ < "1.7":
        torch.jit._enabled = enabled
    else:
        if enabled:
            torch.jit._state.enable()
        else:
            torch.jit._state.disable()


def parse_args():
    set_jit_enabled(False)
    parser = argparse.ArgumentParser(description="Generate video from text prompt using Wan-Diffuser")
    parser.add_argument(
        "--model_id", type=str, default="hunyuanvideo-community/HunyuanVideo", help="Model ID to use for generation"
    )
    parser.add_argument("--prompt", type=str, default=_DEFAULT_PROMPT, help="Text prompt for video generation")
    parser.add_argument("--prompt_file", type=str, default=None, help="Text prompt for video generation")
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=_DEFAULT_NEGATIVE_PROMPT,
        help="Negative text prompt to avoid certain features",
    )
    parser.add_argument("--start-index", type=int, default=-1, help="Start index of the prompt file")
    parser.add_argument("--end-index", type=int, default=-1, help="End index of the prompt file")
    parser.add_argument("--height", type=int, default=720, help="Height of the generated video")
    parser.add_argument("--width", type=int, default=1280, help="Width of the generated video")
    parser.add_argument("--num_frames", type=int, default=129, help="Number of frames in the generated video")
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="Number of denoising steps in the generated video"
    )
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--seed", type=int, default=1024, help="Random seed for generation")
    parser.add_argument("--warmup_step", type=int, default=0, help="Number of warmup runs before the actual generation")
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    parser.add_argument(
        "--strategy",
        type=str,
        default="dense",
        choices=["dense", "usp", "svg", "sparge"],
        help="Strategy to use for generation",
    )
    parser.add_argument("--ulysses-size", type=int, default=None, help="Ulysses size")
    parser.add_argument("--ring-size", type=int, default=None, help="Ring size")
    parser.add_argument("--num_sampled_rows", type=int, default=64, help="Number of sampled rows")
    parser.add_argument("--sparsity", type=float, default=0.25, help="Sparsity of the attention matrix")
    parser.add_argument("--first_layers_fp", type=float, default=0.025, help="First layers fp")
    parser.add_argument("--first_times_fp", type=float, default=0.075, help="First times fp")

    args = parser.parse_args()

    # some sanity checks
    assert args.prompt is not None or args.prompt_file is not None, "Either --prompt or --prompt_file must be provided"
    if args.prompt_file is not None:
        assert Path(args.prompt_file).exists(), f"Prompt file {args.prompt_file} does not exist"
    if args.prompt_file is not None:
        assert not args.profile, "Profile cannot be used with --prompt_file"
    return args


def generate_video(pipe: HunyuanVideoPipeline, args: argparse.Namespace, prompt: str, negative_prompt: str, profiler: torch.profiler.profile = None, strategy: VideoGenStrategy = None):
    if isinstance(strategy, HunyuanSVG):
        _, _, prompt_mask = pipe.encode_prompt(
            prompt,
        )
        prompt_len = prompt_mask.sum()
        strategy.build_attn_masks(pipe.transformer, prompt_len)

        for block in pipe.transformer.transformer_blocks + pipe.transformer.single_transformer_blocks:
            assert block.attn.processor.attention_masks is not None
            assert block.attn.processor.context_length > 0

    if profiler is not None:
        profiler.start()

    if profiler is not None:
        profiler.step()
    
    start = get_time_stamp()

    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        guidance_scale=5.0,
        num_inference_steps=args.num_inference_steps,
        generator=torch.Generator(device="cuda").manual_seed(args.seed),
    ).frames[0]

    end = get_time_stamp()

    if profiler is not None:
        profiler.stop()

    if not is_distributed() or is_rank_zero():
        # Create parent directory for output file if it doesn't exist
        # replace the special character with empty string
        video_name = re.sub(r'[^A-Za-z0-9]', '', prompt)[:40]
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        video_path = output_dir.joinpath(f"{video_name}.mp4")
        export_to_video(output, video_path, fps=15)
        print(f"Video saved to {video_path}")

        # save the log
        log_path = output_dir.joinpath(f"{video_name}.log")
        log_data = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "generation_time": end - start,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "num_inference_steps": args.num_inference_steps,
            "seed": args.seed,
            "strategy": args.strategy,
            "sparsity": args.sparsity,
            "first_layers_fp": args.first_layers_fp,
            "first_times_fp": args.first_times_fp,
            "num_sampled_rows": args.num_sampled_rows,
            "ring_size": args.ring_size,
            "ulysses_size": args.ulysses_size,
            "warmup_step": args.warmup_step,
        }
        with open(log_path, "w") as f:
            # print args as a json string
            json.dump(log_data, f, indent=4)


@torch.no_grad()
def main():
    args = parse_args()

    if is_distributed():
        init_distributed()

    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        args.model_id, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    pipe = HunyuanVideoPipeline.from_pretrained(args.model_id, transformer=transformer, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    pipe.vae.enable_tiling()

    if args.strategy == "dense":
        strategy = HunyuanDense(pipe)
    elif args.strategy == "sparge":
        strategy = HunyuanSparge(pipe)
    elif args.strategy == "usp":
        strategy = HunyuanUsp(pipe, args.ulysses_size, args.ring_size)
    elif args.strategy == "svg":
        strategy = HunyuanSVG(
            pipe,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_sampled_rows=args.num_sampled_rows,
            sample_mse_max_row=10000,
            sparsity=args.sparsity,
            first_layers_fp=args.first_layers_fp,
            first_times_fp=args.first_times_fp,
        )
    else:
        raise ValueError(f"Invalid strategy: {args.strategy}")
    morpher = Morpher(pipe.transformer, strategy)
    morpher.transform()
    print(f"Morphed transformer with {args.strategy} strategy")

    # handle prompts
    if args.prompt_file is not None:
        with open(args.prompt_file, "r") as f:
            prompts = f.readlines()
            prompts = [prompt.strip() for prompt in prompts]

            # remove empty lines
            prompts = [prompt for prompt in prompts if prompt]
        negative_prompts = [None] * len(prompts)
    else:
        prompts = [args.prompt]
        negative_prompts = [args.negative_prompt]

    if args.start_index != -1 and args.end_index != -1:
        prompts = prompts[args.start_index:args.end_index]
        negative_prompts = negative_prompts[args.start_index:args.end_index]

    print(f"Warmup {args.warmup_step} steps")
    for _ in range(args.warmup_step):
        prompt = prompts[0]
        if isinstance(strategy, HunyuanSVG):
            _, _, prompt_mask = pipe.encode_prompt(
                prompt,
            )
            prompt_len = prompt_mask.sum()
            strategy.build_attn_masks(pipe.transformer, prompt_len)

            for block in pipe.transformer.transformer_blocks + pipe.transformer.single_transformer_blocks:
                assert block.attn.processor.attention_masks is not None
                assert block.attn.processor.context_length > 0
        pipe(
            prompt=prompt,
            negative_prompt=None,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            guidance_scale=5.0,
            num_inference_steps=3,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
        ).frames[0]
    

    if args.profile:
        profiler = torch.profiler.profile(
            schedule=torch.profiler.schedule(warmup=args.warmup_step, active=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(Path(args.output_dir, "profiler"))),
            record_shapes=True,
            with_stack=True,
        )
    else:
        profiler = None

    for prompt, negative_prompt in zip(prompts, negative_prompts):
        generate_video(pipe, args, prompt, negative_prompt, profiler, strategy)

    if is_distributed():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
