# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
from datetime import datetime
import logging
import os
import sys
import math
import warnings
import numpy as np

warnings.filterwarnings("ignore")

import torch, random
import torch.distributed as dist
from PIL import Image

import svg.models.wan_orig as wan
from svg.models.wan_orig.configs import (
    WAN_CONFIGS,
    SIZE_CONFIGS,
    MAX_AREA_CONFIGS,
    SUPPORTED_SIZES,
)
from svg.models.wan_orig.utils.prompt_extend import (
    DashScopePromptExpander,
    QwenPromptExpander,
)
from svg.models.wan_orig.utils.utils import cache_video, cache_image, str2bool

EXAMPLE_PROMPT = {
    "t2v-1.3B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2v-14B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2i-14B": {
        "prompt": "一个朴素端庄的美人",
    },
    "i2v-14B": {
        "prompt": "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image": "examples/i2v_input.JPG",
    },
}


def sparsity_to_width(sparsity, context_length, num_frame, frame_size):
    seq_len = context_length + num_frame * frame_size
    total_elements = seq_len**2

    sparsity = (
        sparsity * total_elements - 2 * seq_len * context_length
    ) / total_elements

    width = seq_len * (1 - math.sqrt(1 - sparsity))
    width_frame = width / frame_size

    return width_frame


def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    # The default sampling steps are 40 for image-to-video tasks and 50 for text-to-video tasks.
    if args.sample_steps is None:
        args.sample_steps = 40 if "i2v" in args.task else 50

    if args.sample_shift is None:
        args.sample_shift = 5.0
        if "i2v" in args.task and args.size in ["832*480", "480*832"]:
            args.sample_shift = 3.0

    # The default number of frames are 1 for text-to-image tasks and 81 for other tasks.
    if args.frame_num is None:
        args.frame_num = 1 if "t2i" in args.task else 81

    # T2I frame_num check
    if "t2i" in args.task:
        assert (
            args.frame_num == 1
        ), f"Unsupport frame_num {args.frame_num} for task {args.task}"

    args.base_seed = (
        args.base_seed if args.base_seed >= 0 else random.randint(0, sys.maxsize)
    )

    def seed_everything(seed):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    seed_everything(args.base_seed)

    # Size check
    assert (
        args.size in SUPPORTED_SIZES[args.task]
    ), f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image.",
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames to sample from a image or video. The number should be 4n+1",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.",
    )
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage.",
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.",
    )
    parser.add_argument(
        "--ring_size",
        type=int,
        default=1,
        help="The size of the ring attention parallelism in DiT.",
    )
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.",
    )
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.",
    )
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.",
    )
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated image or video to.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the image or video from.",
    )
    parser.add_argument(
        "--use_prompt_extend",
        action="store_true",
        default=False,
        help="Whether to use prompt extend.",
    )
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use.",
    )
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="The prompt extend model to use.",
    )
    parser.add_argument(
        "--prompt_extend_target_lang",
        type=str,
        default="zh",
        choices=["zh", "en"],
        help="The target language of prompt extend.",
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="The seed to use for generating the image or video.",
    )
    parser.add_argument(
        "--image", type=str, default=None, help="The image to generate the video from."
    )
    parser.add_argument(
        "--sample_solver",
        type=str,
        default="unipc",
        choices=["unipc", "dpm++"],
        help="The solver used to sample.",
    )
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps."
    )
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.",
    )
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=5.0,
        help="Classifier free guidance scale.",
    )

    # Sparse VideoGen
    parser.add_argument(
        "--first_layers_fp",
        type=float,
        default=0.0,
        help="Only works for best config. Leave the 0, 1, 2, 40, 41 layers in FP",
    )
    parser.add_argument(
        "--first_times_fp",
        type=float,
        default=0.0,
        help="Only works for best config. Leave the first 10% timestep in FP",
    )
    parser.add_argument(
        "--record_attention", action="store_true", help="Record the attention"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        required=True,
        choices=["dense", "SVG"],
        help="Sparse Pattern",
    )
    parser.add_argument(
        "--num_sampled_rows", type=int, default=32, help="The number of sampled rows"
    )
    parser.add_argument(
        "--sample_mse_max_row",
        type=int,
        default=10000,
        help="Since some attention masks are really large, need to restrict the maximum size (the row we are going to sample on).",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=1.0,
        help="The sparsity of the striped attention pattern. Accepts one or two float values. Only effective for fast_sample_mse",
    )

    args = parser.parse_args()

    _validate_args(args)

    return args


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)],
        )
    else:
        logging.basicConfig(level=logging.ERROR)


# ============================== Sparse VideoGen ==============================
def svg_replace_wan_attention(wan_pipeline, cfg, args, img=None):
    def print_operator_log_data(module, input, output):
        from svg.timer import operator_log_data, ENABLE_LOGGING, clear_operator_log_data

        if ENABLE_LOGGING:
            max_key_length = max(len(str(key)) for key in operator_log_data.keys())

            formatted_lines = []
            for key, value in dict(sorted(operator_log_data.items())).items():
                formatted_value = f"{value:.2f}"
                line = f"{key:<{max_key_length}} : {formatted_value:>8} ms"
                formatted_lines.append(line)

            print("\n".join(formatted_lines))

            # Renew every print
            clear_operator_log_data()

    wan_pipeline.model.register_forward_hook(print_operator_log_data)

    # Get the config
    cfg_size, num_head, head_dim, dtype, device = (
        1,
        cfg["num_layers"],
        cfg["dim"] // cfg["num_heads"],
        torch.bfloat16,
        "cuda",
    )  # Actually it has cfg, but the pipeline will inference it separately
    context_length, num_frame = (
        0,
        (args.frame_num - 1) // wan_pipeline.vae_stride[0] + 1,
    )

    if img is None:  # T2V
        frame_size = (SIZE_CONFIGS[args.size][1] // cfg["vae_stride"][1]) * (
            SIZE_CONFIGS[args.size][0] // cfg["vae_stride"][2]
        )
        frame_size = frame_size // (cfg["patch_size"][1] * cfg["patch_size"][2])
    else:  # I2V
        max_area = 720 * 1280
        
        import torchvision.transforms.functional as TF
        img = TF.to_tensor(img).sub_(0.5).div_(0.5)
        h, w = img.shape[1:]
        aspect_ratio = h / w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio)
            // cfg["vae_stride"][1]
            // cfg["patch_size"][1]
            * cfg["patch_size"][1]
        )
        lat_w = round(
            np.sqrt(max_area / aspect_ratio)
            // cfg["vae_stride"][2]
            // cfg["patch_size"][2]
            * cfg["patch_size"][2]
        )
        frame_size = lat_h * lat_w // (cfg["patch_size"][1] * cfg["patch_size"][2])
  
    # Sparsity to width
    spatial_width = temporal_width = sparsity_to_width(
        args.sparsity, context_length, num_frame, frame_size
    )

    print(
        f"Spatial_width: {spatial_width}, Temporal_width: {temporal_width}. Sparsity: {args.sparsity}"
    )

    if args.pattern == "SVG":
        masks = ["spatial", "temporal"]

        def get_attention_mask(mask_name):

            from termcolor import colored

            allocated = torch.cuda.memory_allocated() / 1e9
            print(colored(f"Allocated Memory: {allocated:.2f} GB", "yellow"))

            attention_mask = torch.zeros(
                (
                    context_length + num_frame * frame_size,
                    context_length + num_frame * frame_size,
                ),
                device="cpu",
            )

            # TODO: fix hard coded mask
            if mask_name == "spatial":
                pixel_attn_mask = torch.zeros_like(
                    attention_mask, dtype=torch.bool, device="cpu"
                )
                block_size, block_thres = 128, frame_size * 1.5
                num_block = math.ceil(num_frame * frame_size / block_size)
                for i in range(num_block):
                    for j in range(num_block):
                        if abs(i - j) < block_thres // block_size:
                            pixel_attn_mask[
                                i * block_size : (i + 1) * block_size,
                                j * block_size : (j + 1) * block_size,
                            ] = 1
                attention_mask = pixel_attn_mask
            else:
                pixel_attn_mask = torch.zeros_like(
                    attention_mask, dtype=torch.bool, device=device
                )

                block_size, block_thres = 128, frame_size * 1.5
                num_block = math.ceil(num_frame * frame_size / block_size)
                for i in range(num_block):
                    for j in range(num_block):
                        if abs(i - j) < block_thres // block_size:
                            pixel_attn_mask[
                                i * block_size : (i + 1) * block_size,
                                j * block_size : (j + 1) * block_size,
                            ] = 1

                pixel_attn_mask = (
                    pixel_attn_mask.reshape(
                        frame_size, num_frame, frame_size, num_frame
                    )
                    .permute(1, 0, 3, 2)
                    .reshape(frame_size * num_frame, frame_size * num_frame)
                )
                attention_mask = pixel_attn_mask

            attention_mask = attention_mask[: args.sample_mse_max_row].cuda()
            return attention_mask

        from svg.models.wan_orig.modules.attention import (
            Wan_SparseAttn,
            prepare_flexattention,
            prepare_dense_attention,
        )
        from svg.models.wan_orig.modules.custom_model import replace_sparse_forward

        AttnModule = Wan_SparseAttn
        AttnModule.num_sampled_rows = args.num_sampled_rows
        AttnModule.sample_mse_max_row = args.sample_mse_max_row
        AttnModule.attention_masks = [
            get_attention_mask(mask_name) for mask_name in masks
        ]
        AttnModule.first_layers_fp = args.first_layers_fp
        AttnModule.first_times_fp = args.first_times_fp
        AttnModule.context_length = context_length
        AttnModule.num_frame = num_frame
        AttnModule.frame_size = frame_size

        dense_block_mask = prepare_dense_attention(
            cfg_size,
            num_head,
            head_dim,
            dtype,
            device,
            context_length,
            context_length,
            num_frame,
            frame_size,
        )
        AttnModule.dense_block_mask = dense_block_mask

        block_mask = prepare_flexattention(
            cfg_size,
            num_head,
            head_dim,
            dtype,
            device,
            context_length,
            context_length,
            num_frame,
            frame_size,
            diag_width=spatial_width,
            multiplier=temporal_width,
        )
        AttnModule.block_mask = block_mask
        print(AttnModule.block_mask)

        AttnModule.record_attention = args.record_attention
        if args.record_attention:
            # create file to record the attention
            save_dir = "result"
            os.makedirs(save_dir, exist_ok=True)
            record_path = os.path.join(
                save_dir, f"record_attention-{args.base_seed}.jsonl"
            )
            with open(record_path, "w"):
                pass
            AttnModule.record_path = record_path

        replace_sparse_forward()
        

    # Convert linear weights to the specified parameter dtype before moving to device
    for name, module in wan_pipeline.model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if hasattr(module, 'weight') and module.weight is not None:
                module.weight.data = module.weight.data.to(cfg.param_dtype)

    # ============================== Sparse VideoGen Ends ==============================
    return wan_pipeline


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl", init_method="env://", rank=rank, world_size=world_size
        )
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1 or args.ring_size > 1
        ), f"context parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1 or args.ring_size > 1:
        assert (
            args.ulysses_size * args.ring_size == world_size
        ), f"The number of ulysses_size and ring_size should be equal to the world size."
        from xfuser.core.distributed import (
            initialize_model_parallel,
            init_distributed_environment,
        )

        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size()
        )

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size,
        )

    if args.use_prompt_extend:
        if args.prompt_extend_method == "dashscope":
            prompt_expander = DashScopePromptExpander(
                model_name=args.prompt_extend_model, is_vl="i2v" in args.task
            )
        elif args.prompt_extend_method == "local_qwen":
            prompt_expander = QwenPromptExpander(
                model_name=args.prompt_extend_model,
                is_vl="i2v" in args.task,
                device=rank,
            )
        else:
            raise NotImplementedError(
                f"Unsupport prompt_extend_method: {args.prompt_extend_method}"
            )

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert (
            cfg.num_heads % args.ulysses_size == 0
        ), f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    if "t2v" in args.task or "t2i" in args.task:
        if args.prompt is None:
            args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
        logging.info(f"Input prompt: {args.prompt}")
        if args.use_prompt_extend:
            logging.info("Extending prompt ...")
            if rank == 0:
                prompt_output = prompt_expander(
                    args.prompt,
                    tar_lang=args.prompt_extend_target_lang,
                    seed=args.base_seed,
                )
                if prompt_output.status == False:
                    logging.info(f"Extending prompt failed: {prompt_output.message}")
                    logging.info("Falling back to original prompt.")
                    input_prompt = args.prompt
                else:
                    input_prompt = prompt_output.prompt
                input_prompt = [input_prompt]
            else:
                input_prompt = [None]
            if dist.is_initialized():
                dist.broadcast_object_list(input_prompt, src=0)
            args.prompt = input_prompt[0]
            logging.info(f"Extended prompt: {args.prompt}")

        logging.info("Creating WanT2V pipeline.")
        wan_t2v = wan.WanT2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
        )

        wan_t2v = svg_replace_wan_attention(wan_t2v, cfg, args)

        logging.info(f"Generating {'image' if 't2i' in args.task else 'video'} ...")
        video = wan_t2v.generate(
            args.prompt,
            size=SIZE_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model,
        )

    else:
        if args.prompt is None:
            args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
        if args.image is None:
            args.image = EXAMPLE_PROMPT[args.task]["image"]
        logging.info(f"Input prompt: {args.prompt}")
        logging.info(f"Input image: {args.image}")

        img = Image.open(args.image).convert("RGB")
        if args.use_prompt_extend:
            logging.info("Extending prompt ...")
            if rank == 0:
                prompt_output = prompt_expander(
                    args.prompt,
                    tar_lang=args.prompt_extend_target_lang,
                    image=img,
                    seed=args.base_seed,
                )
                if prompt_output.status == False:
                    logging.info(f"Extending prompt failed: {prompt_output.message}")
                    logging.info("Falling back to original prompt.")
                    input_prompt = args.prompt
                else:
                    input_prompt = prompt_output.prompt
                input_prompt = [input_prompt]
            else:
                input_prompt = [None]
            if dist.is_initialized():
                dist.broadcast_object_list(input_prompt, src=0)
            args.prompt = input_prompt[0]
            logging.info(f"Extended prompt: {args.prompt}")

        logging.info("Creating WanI2V pipeline.")
        wan_i2v = wan.WanI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
        )

        wan_i2v = svg_replace_wan_attention(wan_i2v, cfg, args, img=img)

        logging.info("Generating video ...")
        video = wan_i2v.generate(
            args.prompt,
            img,
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model,
        )

    if rank == 0:
        if args.save_file is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = args.prompt.replace(" ", "_").replace("/", "_")[:50]
            suffix = ".png" if "t2i" in args.task else ".mp4"
            args.save_file = (
                f"{args.task}_{args.size.replace('*','x') if sys.platform=='win32' else args.size}"
                + suffix
            )

        if "t2i" in args.task:
            logging.info(f"Saving generated image to {args.save_file}")
            cache_image(
                tensor=video.squeeze(1)[None],
                save_file=args.save_file,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )
        else:
            logging.info(f"Saving generated video to {args.save_file}")
            cache_video(
                tensor=video[None],
                save_file=args.save_file,
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )
    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()

    # Due to some strange problem on my machine
    torch.backends.cuda.preferred_linalg_library(backend="magma")
    generate(args)
