import argparse
import json
import os
from glob import glob
import math
from copy import deepcopy

from termcolor import colored
import numpy as np
import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel, FlowMatchEulerDiscreteScheduler
from diffusers.utils import load_image, export_to_video

from dataloader import load_prompt_or_image
from svg.timer import print_operator_log_data
from svg.utils.seed import seed_everything
from svg.models.hyvideo.inference import replace_hyvideo_flashattention, replace_hyvideo_attention
from svg.models.hyvideo.utils import get_prompt_length

from svg.logger import logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate video from text prompt using Wan-Diffuser")
    parser.add_argument("--model_id", type=str, default="tencent/HunyuanVideo", help="Model ID to use for generation")
    parser.add_argument("--data_path", type=str, default=None, help="Path of VBench I2V data suite")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt for video generation")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Negative text prompt to avoid certain features")

    parser.add_argument("--prompt_source", type=str, default="prompt", choices=["prompt", "T2V_Hyv_VBench", "T2V_Hyv_Web", "T2V_Xingyang_Motion", "T2V_Xingyang_VBench"], help="Source of the prompt")
    parser.add_argument("--prompt_idx", type=int, default=0, help="Index of the prompt")

    parser.add_argument("--height", type=int, default=720, help="Height of the generated video")
    parser.add_argument("--width", type=int, default=1280, help="Width of the generated video")
    parser.add_argument("--num_frames", type=int, default=129, help="Number of frames in the generated video")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps in the generated video")
    parser.add_argument("--resolution", type=str, default="720p", choices=["480p", "720p"], help="Resolution of the generated video")
    parser.add_argument("--output_file", type=str, default="output.mp4", help="Output video file name")
    parser.add_argument("--logging_file", type=str, default=None, help="Path to the logging file.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for generation")
    parser.add_argument("--skip_existing", action="store_true", help="Skip generating existing output files")

    parser.add_argument("--pattern", type=str, default="dense", choices=["SVG", "dense", "SAP"])
    parser.add_argument("--first_layers_fp", type=float, default=0.025, help="Only works for best config. Leave the 0, 1, 2, 40, 41 layers in FP")
    parser.add_argument("--first_times_fp", type=float, default=0.075, help="Only works for best config. Leave the first 10% timestep in FP")

    # SVG1 specific
    parser.add_argument("--num_sampled_rows", type=int, default=64, help="The number of sampled rows")
    parser.add_argument("--sample_mse_max_row", type=int, default=10000, help="The maximum number of rows in attention mask. Prevent OOM.")
    parser.add_argument("--sparsity", type=float, default=0.25, help="The sparsity of the striped attention pattern. Accepts one or two float values.")

    # SVG2 specific
    parser.add_argument("--num_q_centroids", "--qc", type=int, default=50, help="Number of query centroids for SAP.")
    parser.add_argument("--num_k_centroids", "--kc", type=int, default=200, help="Number of key centroids for SAP.")
    parser.add_argument("--top_p_kmeans", type=float, default=0.9, help="Top-p threshold for block selection in SAP.")
    parser.add_argument("--min_kc_ratio", type=float, default=0, help="At least this proportion of key blocks to keep per query block in SAP.")
    parser.add_argument("--kmeans_iter_init", type=int, default=0, help="Number of KMeans iterations for initialization in SAP.")
    parser.add_argument("--kmeans_iter_step", type=int, default=0, help="Number of KMeans iterations for other diffusion steps in SAP.")
    parser.add_argument("--zero_step_kmeans_init", action="store_true", help="Initialize the centroids for the first step in SAP, not after warmup.")

    args = parser.parse_args()

    seed_everything(args.seed)

    # In some cases it will raise RuntimeError: cusolver error: CUSOLVER_STATUS_INTERNAL_ERROR
    torch.backends.cuda.preferred_linalg_library(backend="magma")
    
    if args.skip_existing:
        if os.path.exists(args.output_file):
            logger.info(f"Output file {args.output_file} already exists. Skipping generation.")
            exit(0)

    #########################################################
    # Load the model
    #########################################################
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        args.model_id, subfolder="transformer", torch_dtype=torch.bfloat16, revision='refs/pr/18'
    )
    flow_shift = 7.0
    scheduler = FlowMatchEulerDiscreteScheduler(shift=flow_shift)
    pipe = HunyuanVideoPipeline.from_pretrained(args.model_id, transformer=transformer, scheduler=scheduler, revision='refs/pr/18', torch_dtype=torch.bfloat16)
    pipe.vae.enable_tiling()
    pipe.to("cuda")
    
    config = pipe.transformer.config

    #########################################################
    # Translate the percentage of warmup of layers and timesteps to the actual layers and timesteps
    #########################################################
    ref_scheduler = deepcopy(pipe.scheduler)
    ref_scheduler.set_timesteps(args.num_inference_steps)
    ref_timesteps = ref_scheduler.timesteps
    total_layers = config.num_layers + config.num_single_layers
    
    num_fp_timesteps = math.floor(args.first_times_fp * args.num_inference_steps)
    num_fp_layers = math.floor(args.first_layers_fp * total_layers)
    if num_fp_timesteps > 0:
        args.first_times_fp = ref_scheduler.timesteps[num_fp_timesteps - 1] - 1
    else:
        args.first_times_fp = 1001 # 1000 is the first timestep
    args.first_layers_fp = num_fp_layers
    
    logger.info(f"Warmup of Timesteps: {num_fp_timesteps} / {args.num_inference_steps} || {args.first_times_fp} / 1000 use FP")
    logger.info(f"Warmup of Layers: {num_fp_layers} / {total_layers} use FP")
    
    #########################################################
    # Load the prompt and image path
    #########################################################
    args.prompt, _ = load_prompt_or_image(args.prompt_source, args.prompt_idx, args.prompt, None)

    if args.prompt is None:
        print(colored("Using default prompt", "red"))
        args.prompt = "A cat walks on the grass, realistic"

    if args.negative_prompt is None:
        args.negative_prompt = "Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion"
    
    prompt_length = get_prompt_length(pipe, args.prompt)
    print(f"Prompt length: {prompt_length}")

    #########################################################
    # Replace the attention
    #########################################################
    replace_hyvideo_flashattention(pipe)

    if args.pattern == "SVG":
        replace_hyvideo_attention(
            pipe,
            args.height,
            args.width,
            args.num_frames,
            prompt_length,
            first_layers_fp=args.first_layers_fp,
            first_times_fp=args.first_times_fp,
            pattern=args.pattern,
            # SVG specific
            num_sampled_rows=args.num_sampled_rows,
            sample_mse_max_row=args.sample_mse_max_row,
            sparsity=args.sparsity,
        )
    elif args.pattern == "SAP":
        replace_hyvideo_attention(
            pipe,
            args.height,
            args.width,
            args.num_frames,
            prompt_length,
            first_layers_fp=args.first_layers_fp,
            first_times_fp=args.first_times_fp,
            pattern=args.pattern,
            # SAP specific
            num_q_centroids=args.num_q_centroids,
            num_k_centroids=args.num_k_centroids,
            top_p_kmeans=args.top_p_kmeans,
            min_kc_ratio=args.min_kc_ratio,
            logging_file=args.logging_file,
            kmeans_iter_init=args.kmeans_iter_init,
            kmeans_iter_step=args.kmeans_iter_step,
            zero_step_kmeans_init=args.zero_step_kmeans_init,
        )
    else:
        assert args.pattern == "dense", f"Invalid pattern: {args.pattern}"
        
    # Print time logger
    for layer_idx, block in enumerate(pipe.transformer.transformer_blocks):
        block.register_forward_hook(print_operator_log_data)
    for layer_idx, block in enumerate(pipe.transformer.single_transformer_blocks):
        block.register_forward_hook(print_operator_log_data)

    #########################################################
    # Generate the video
    #########################################################
    output = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        guidance_scale=6.0,
        num_inference_steps=args.num_inference_steps,
    ).frames[0]

    # Create parent directory for output file if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    export_to_video(output, args.output_file, fps=24)
