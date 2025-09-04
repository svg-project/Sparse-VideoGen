import argparse
import json
import os
from glob import glob

import numpy as np
import torch
from diffusers import HunyuanVideoImageToVideoPipeline, HunyuanVideoTransformer3DModel, FlowMatchEulerDiscreteScheduler
from diffusers.utils import load_image, export_to_video

from dataloader import load_prompt_or_image
from svg.timer import print_operator_log_data
from svg.utils.seed import seed_everything
from svg.models.hyvideo.attention import replace_hyvideo_flashattention

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate video from text prompt using Wan-Diffuser")
    parser.add_argument("--model_id", type=str, default="hunyuanvideo-community/HunyuanVideo-I2V", help="Model ID to use for generation")
    parser.add_argument("--data_path", type=str, default=None, help="Path of VBench I2V data suite")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt for video generation")
    parser.add_argument("--image_path", type=str, default=None, help="Path of image")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Negative text prompt to avoid certain features")

    parser.add_argument("--prompt_source", type=str, default="prompt", choices=["prompt", "I2V_VBench", "I2V_Wan_Web"], help="Source of the prompt")
    parser.add_argument("--prompt_idx", type=int, default=0, help="Index of the prompt")

    parser.add_argument("--num_frames", type=int, default=129, help="Number of frames in the generated video")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps in the generated video")
    parser.add_argument("--resolution", type=str, default="720p", choices=["480p", "720p"], help="Resolution of the generated video")
    parser.add_argument("--output_file", type=str, default="output.mp4", help="Output video file name")
    parser.add_argument("--logging_file", type=str, default=None, help="Path to the logging file.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for generation")

    parser.add_argument("--pattern", type=str, default="dense", choices=["dense", "sparse"], help="Pattern of the generated video")

    args = parser.parse_args()

    seed_everything(args.seed)

    # In some cases it will raise RuntimeError: cusolver error: CUSOLVER_STATUS_INTERNAL_ERROR
    torch.backends.cuda.preferred_linalg_library(backend="magma")

    #########################################################
    # Load the model
    #########################################################
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(args.model_id, subfolder="transformer", torch_dtype=torch.bfloat16)
    flow_shift = 7.0
    scheduler = FlowMatchEulerDiscreteScheduler(shift=flow_shift)
    pipe = HunyuanVideoImageToVideoPipeline.from_pretrained(args.model_id, transformer=transformer, scheduler=scheduler, torch_dtype=torch.bfloat16)
    pipe.vae.enable_tiling()
    pipe.to("cuda")

    #########################################################
    # Load the prompt and image path
    #########################################################
    args.prompt, args.image_path = load_prompt_or_image(args.prompt_source, args.prompt_idx, args.prompt, args.image_path)

    if args.prompt is not None:
        assert args.image_path is not None, "Image path must be provided"
        image = load_image(args.image_path)

        # Reshape
        max_area = 720 * 1280 if args.resolution == "720p" else 544 * 960
        aspect_ratio = image.height / image.width
        mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size
        args.height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        args.width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        image = image.resize((args.width, args.height))

    else:
        raise ValueError("Prompt must be provided")

    if args.negative_prompt is None:
        args.negative_prompt = "Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion"

    #########################################################
    # Replace the attention
    #########################################################
    replace_hyvideo_flashattention(pipe)

    #########################################################
    # Generate the video
    #########################################################
    output = pipe(
        image=image,
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
