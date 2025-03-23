import os
import json
from termcolor import colored

import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
import argparse

from svg.utils import seed_everything
from svg.models.wan.inference import replace_wan_attention

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate video from text prompt using Wan-Diffuser")
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-T2V-14B-Diffusers", help="Model ID to use for generation")
    parser.add_argument("--file_idx", type=int, default=None, help="Index of prompt")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt for video generation")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Negative text prompt to avoid certain features")
    parser.add_argument("--height", type=int, default=720, help="Height of the generated video")
    parser.add_argument("--width", type=int, default=1280, help="Width of the generated video")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames in the generated video")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps in the generated video")
    parser.add_argument("--output_file", type=str, default="output.mp4", help="Output video file name")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for generation")

    parser.add_argument("--pattern", type=str, default="dense", choices=["SVG", "dense"])
    parser.add_argument("--first_layers_fp", type=float, default=0.025, help="Only works for best config. Leave the 0, 1, 2, 40, 41 layers in FP")
    parser.add_argument("--first_times_fp", type=float, default=0.075, help="Only works for best config. Leave the first 10% timestep in FP")
    parser.add_argument("--num_sampled_rows", type=int, default=64, help="The number of sampled rows")
    parser.add_argument("--sample_mse_max_row", type=int, default=10000, help="The maximum number of rows in attention mask. Prevent OOM.")
    parser.add_argument("--sparsity", type=float, default=0.25, help="The sparsity of the striped attention pattern. Accepts one or two float values.")
    args = parser.parse_args()
    
    seed_everything(args.seed)
    
    # Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
    model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    flow_shift = 5.0 # 5.0 for 720P, 3.0 for 480P
    scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift)
    pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
    pipe.scheduler = scheduler
    pipe.to("cuda")

    if args.prompt is None:
        if args.file_idx:
            data = json.load(open(f"captions_sora.json"))
            args.prompt = data[args.file_idx]

            output_dir = f"result/{args.pattern}"
            os.makedirs(output_dir, exist_ok=True)
            # Define the output file path
            args.output_file = f"{output_dir}/{args.file_idx}-{args.seed}.mp4"
        else:
            print(colored("Using default prompt", "red"))
            args.prompt = "A cat walks on the grass, realistic"
    
    if args.negative_prompt is None:
        args.negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

    print("=" * 20 + " Prompts " + "=" * 20)
    print(f"Prompt: {args.prompt}\n\n" + f"Negative Prompt: {args.negative_prompt}")

    if args.pattern == "SVG":
        replace_wan_attention(
            pipe,
            args.height,
            args.width,
            args.num_frames,
            args.num_sampled_rows,
            args.sample_mse_max_row,
            args.sparsity,
            args.first_layers_fp,
            args.first_times_fp
        )
        
    output = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        guidance_scale=5.0,
        num_inference_steps=args.num_inference_steps
    ).frames[0]
    
    
    export_to_video(output, args.output_file, fps=16)
