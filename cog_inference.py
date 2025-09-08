import argparse

import torch
from diffusers import CogVideoXImageToVideoPipeline

from svg.models.cog.utils import seed_everything
from svg.models.cog.inference import replace_cog_attention, sample_image
import os

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
    # Parallel inference parameters
    parser.add_argument(
        "--use_sequence_parallel",
        action="store_true",
        help="Enable sequence parallelism for parallel inference"
    )
    parser.add_argument(
        "--ulysses_degree",
        type=int,
        default=2,
        help="The number of ulysses parallel"
    )
    
    args = parser.parse_args()

    if args.use_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import init_distributed_environment, initialize_model_parallel
        
        # Setup distributed environment
        rank = int(os.getenv("RANK", 0))
        world_size = int(os.getenv("WORLD_SIZE", 1))
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        
        assert world_size > 1, f"Sequence parallelism requires world_size > 1, got {world_size}"
        assert args.ulysses_degree > 1, "ulysses_degree must be > 1 for sequence parallelism"
        assert world_size == args.ulysses_degree, (
            f"Currently only pure Ulysses parallelism is supported. "
            f"world_size ({world_size}) must equal ulysses_degree ({args.ulysses_degree})"
        )
        
        # Initialize PyTorch distributed
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
        
        # Initialize xFuser model parallelism
        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=1,
            ulysses_degree=args.ulysses_degree,
        )
        
        device = local_rank

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