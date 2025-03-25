import os
import json
import argparse
from glob import glob

import torch
import numpy as np
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

from transformers import CLIPVisionModel
from svg.utils import seed_everything

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate video from text prompt using Wan-Diffuser")
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-T2V-14B-Diffusers", help="Model ID to use for generation")
    parser.add_argument("--data_path", type=str, default=None, help="Path of VBench I2V data suite")
    parser.add_argument("--file_idx", type=int, default=None, help="Index of prompt")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt for video generation")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Negative text prompt to avoid certain features")
    parser.add_argument("--height", type=int, default=720, help="Height of the generated video")
    parser.add_argument("--width", type=int, default=1280, help="Width of the generated video")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames in the generated video")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps in the generated video")
    parser.add_argument("--output_file", type=str, default="output.mp4", help="Output video file name")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for generation")

    args = parser.parse_args()
    
    seed_everything(args.seed)
    
    # Available models: Wan-AI/Wan2.1-I2V-14B-480P-Diffusers, Wan-AI/Wan2.1-I2V-14B-720P-Diffusers
    model_id = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
    image_encoder = CLIPVisionModel.from_pretrained(model_id, subfolder="image_encoder", torch_dtype=torch.float32)
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    flow_shift = 5.0 # 5.0 for 720P, 3.0 for 480P
    scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift)
    pipe = WanImageToVideoPipeline.from_pretrained(model_id, vae=vae, image_encoder=image_encoder, torch_dtype=torch.bfloat16)
    pipe.scheduler = scheduler
    pipe.to("cuda")

    if args.prompt is None:
        if args.file_idx:
            # Get the prompt and image
            DATA_PATH = os.path.join(args.data_path, "target_crop", "16-9")
            SOURCE_FILES = sorted(glob(os.path.join(DATA_PATH, "*.jpg")))

            improved_prompt_path = os.path.join(args.data_path, 'target_crop/improved_prompts.json')
            with open(improved_prompt_path, 'r') as file:
                improved_prompts = json.load(file)

            origin_prompt = improved_prompts[str(args.file_idx)]["original"]
            prompt = improved_prompts[str(args.file_idx)]["original"] # ["improved"]
            image_name_or_path = [s for s in SOURCE_FILES if origin_prompt in s][0]
            image = load_image(image_name_or_path)

            # Define the output file path
            output_dir = f"result/dense"
            os.makedirs(output_dir, exist_ok=True)
            args.output_file = f"{output_dir}/{args.file_idx}-{args.seed}.mp4"
        else:
            image = load_image(
                "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
            )
            max_area = 720 * 1280
            aspect_ratio = image.height / image.width
            mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
            height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
            width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
            image = image.resize((width, height))
            prompt = (
                "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in "
                "the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot."
            )
    
    if args.negative_prompt is None:
        negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

    output = pipe(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        guidance_scale=5.0,
        num_inference_steps=args.num_inference_steps
    ).frames[0]
    export_to_video(output, args.output_file, fps=16)
