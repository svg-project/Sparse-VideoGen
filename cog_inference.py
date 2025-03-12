import argparse

import torch
from diffusers import CogVideoXImageToVideoPipeline

from svg.models.cog.utils import seed_everything
from svg.models.cog.inference import replace_cog_attention, sample_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script that sets a random seed.")
    parser.add_argument("--version", type=str, default="v1.5", choices=["v1", "v1.5"], help="Random seed for reproducibility")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--image_path", type=str, required=True, help="Image Path")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt")
    parser.add_argument("--pattern", type=str, default="SVG", choices=["SVG", "dense"])
    parser.add_argument("--num_step", type=int, default=50, help="Number of steps to inference")
    parser.add_argument("--first_layers_fp", type=float, default=0.025, help="Only works for best config. Leave the 0, 1, 2, 40, 41 layers in FP")
    parser.add_argument("--first_times_fp", type=float, default=0.2, help="Only works for best config. Leave the first 10% timestep in FP")
    parser.add_argument("--num_sampled_rows", type=int, default=32, help="The number of sampled rows")
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.25,
        help="The sparsity of the striped attention pattern. Accepts one or two float values. Only effective for fast_sample_mse"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output generated videos"
    )

    args = parser.parse_args()

    seed_everything(args.seed)


    model_id = "THUDM/CogVideoX1.5-5B-I2V"

    dtype = torch.bfloat16

    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype
    ).to("cuda")

    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()

    if args.pattern == "SVG":
        replace_cog_attention(
            pipe,
            args.version,
            args.num_sampled_rows,
            args.sparsity,
            args.first_layers_fp,
            args.first_times_fp
        )
    
    sample_image(
        pipe,
        args.prompt,
        args.image_path,
        args.output_path,
        args.seed,
        args.version,
        args.num_step
    )