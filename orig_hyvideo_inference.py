import json
import math
import os
import time
from datetime import datetime
from pathlib import Path

import torch
from loguru import logger

from dataloader import load_prompt_or_image
from svg.models.hyvideo_orig.config import parse_args
from svg.models.hyvideo_orig.inference import HunyuanVideoSampler
from svg.models.hyvideo_orig.utils.file_utils import save_videos_grid
from svg.timer import print_operator_log_data


def sparsity_to_width(sparsity, context_length, num_frame, frame_size):
    seq_len = context_length + num_frame * frame_size
    total_elements = seq_len**2

    sparsity = (sparsity * total_elements - 2 * seq_len * context_length) / total_elements

    width = seq_len * (1 - math.sqrt(1 - sparsity))
    width_frame = width / frame_size

    return width_frame


if __name__ == "__main__":
    args = parse_args()
    print(args)
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")

    # In some cases it will raise RuntimeError: cusolver error: CUSOLVER_STATUS_INTERNAL_ERROR
    torch.backends.cuda.preferred_linalg_library(backend="magma")

    # Load models
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)

    # Get the updated args
    args = hunyuan_video_sampler.args

    # Sparsity Related
    transformer = hunyuan_video_sampler.pipeline.transformer
    for _, block in enumerate(transformer.double_blocks):
        block.sparse_args = args
    for _, block in enumerate(transformer.single_blocks):
        block.sparse_args = args
    transformer.sparse_args = args

    # Update the prompt
    args.prompt, _ = load_prompt_or_image(args.prompt_source, args.prompt_idx, args.prompt, None)

    # We need to get the prompt len in advance, since HunyuanVideo handle the attention mask in a special way
    prompt_mask = hunyuan_video_sampler.get_prompt_mask(
        prompt=args.prompt,
        height=args.video_size[0],
        width=args.video_size[1],
        video_length=args.video_length,
        negative_prompt=args.neg_prompt,
        infer_steps=args.infer_steps,
        guidance_scale=args.cfg_scale,
        num_videos_per_prompt=args.num_videos,
        embedded_guidance_scale=args.embedded_cfg_scale,
    )
    prompt_len = prompt_mask.sum()

    print(f"Memory: {torch.cuda.memory_allocated() // 1024 ** 2} / {torch.cuda.max_memory_allocated() // 1024 ** 2} MB before Inference")

    cfg_size, num_head, head_dim, dtype, device = 1, 24, 128, torch.bfloat16, "cuda"
    context_length, num_frame = 256, 1 + args.video_length // 4  # TODO: Make it more formal
    frame_size = args.video_size[0] * args.video_size[1] // 256  # TODO: Make it more formal

    save_path = args.output_file

    if args.pattern == "SVG":
        masks = ["spatial", "temporal"]

        # Calculation
        spatial_width = temporal_width = sparsity_to_width(args.sparsity, context_length, num_frame, frame_size)

        print(f"Spatial_width: {spatial_width}, Temporal_width: {temporal_width}. Sparsity: {args.sparsity}")

        def get_attention_mask(mask_name):

            attention_mask = torch.zeros((context_length + num_frame * frame_size, context_length + num_frame * frame_size), device="cpu")

            # TODO: fix hard coded mask
            if mask_name == "spatial":
                pixel_attn_mask = torch.zeros_like(attention_mask[:-context_length, :-context_length], dtype=torch.bool, device="cpu")
                block_size, block_thres = 128, frame_size * 1.5
                num_block = math.ceil(num_frame * frame_size / block_size)
                for i in range(num_block):
                    for j in range(num_block):
                        if abs(i - j) < block_thres // block_size:
                            pixel_attn_mask[i * block_size : (i + 1) * block_size, j * block_size : (j + 1) * block_size] = 1
                attention_mask[:-context_length, :-context_length] = pixel_attn_mask

                attention_mask[-context_length:, :] = 1
                attention_mask[:, -context_length:] = 1
                # attention_mask = torch.load(f"/data/home/xihaocheng/andy_develop/tmp_data/hunyuanvideo/I2VSparse/sparseattn/v5/mask_tensor/mask_spatial.pt", map_location="cpu")

            else:
                pixel_attn_mask = torch.zeros_like(attention_mask[:-context_length, :-context_length], dtype=torch.bool, device=device)

                block_size, block_thres = 128, frame_size * 1.5
                num_block = math.ceil(num_frame * frame_size / block_size)
                for i in range(num_block):
                    for j in range(num_block):
                        if abs(i - j) < block_thres // block_size:
                            pixel_attn_mask[i * block_size : (i + 1) * block_size, j * block_size : (j + 1) * block_size] = 1

                pixel_attn_mask = pixel_attn_mask.reshape(frame_size, num_frame, frame_size, num_frame).permute(1, 0, 3, 2).reshape(frame_size * num_frame, frame_size * num_frame)
                attention_mask[:-context_length, :-context_length] = pixel_attn_mask

                attention_mask[-context_length:, :] = 1
                attention_mask[:, -context_length:] = 1
                # attention_mask = torch.load(f"/data/home/xihaocheng/andy_develop/tmp_data/hunyuanvideo/I2VSparse/sparseattn/v5/mask_tensor/mask_temporal.pt", map_location="cpu")
            attention_mask = attention_mask[: args.sample_mse_max_row].cuda()
            return attention_mask

        from svg.models.hyvideo_orig.modules.attenion import (
            Hunyuan_SparseAttn,
            prepare_flexattention,
        )
        from svg.models.hyvideo_orig.modules.custom_models import replace_sparse_forward

        AttnModule = Hunyuan_SparseAttn

        # These might be needed by the processor if it has to adapt to sequence dimensions
        AttnModule.prompt_length = prompt_len
        AttnModule.context_length = context_length
        AttnModule.num_frame = num_frame
        AttnModule.frame_size = frame_size

        AttnModule.num_sampled_rows = args.num_sampled_rows
        AttnModule.sample_mse_max_row = args.sample_mse_max_row
        AttnModule.attention_masks = [get_attention_mask(mask_name) for mask_name in masks]
        AttnModule.first_layers_fp = args.first_layers_fp
        AttnModule.first_times_fp = args.first_times_fp

        block_mask = prepare_flexattention(cfg_size, num_head, head_dim, dtype, device, context_length, prompt_len, num_frame, frame_size, diag_width=spatial_width, multiplier=temporal_width)
        AttnModule.block_mask = block_mask
        replace_sparse_forward()

        print(block_mask)

    elif args.pattern in ["SAP"]:

        # Make dir and clear the logging file
        if args.logging_file is not None:
            os.makedirs(os.path.dirname(args.logging_file), exist_ok=True)
            with open(args.logging_file, "w") as f:
                f.write("")

        from svg.models.hyvideo_orig.modules.attenion import Hunyuan_SAPAttn
        from svg.models.hyvideo_orig.modules.custom_models import replace_sparse_forward

        AttnModule = Hunyuan_SAPAttn

        AttnModule.first_layers_fp = args.first_layers_fp
        AttnModule.first_times_fp = args.first_times_fp
        AttnModule.logging_file = args.logging_file

        # These might be needed by the processor if it has to adapt to sequence dimensions
        AttnModule.prompt_length = prompt_len
        AttnModule.context_length = context_length
        AttnModule.num_frame = num_frame
        AttnModule.frame_size = frame_size

        AttnModule.num_q_centroids = args.num_q_centroids
        AttnModule.num_k_centroids = args.num_k_centroids
        AttnModule.top_p_kmeans = args.top_p_kmeans
        AttnModule.min_kc_ratio = args.min_kc_ratio
        AttnModule.kmeans_iter_init = args.kmeans_iter_init
        AttnModule.kmeans_iter_step = args.kmeans_iter_step
        AttnModule.zero_step_kmeans_init = args.zero_step_kmeans_init

        replace_sparse_forward()
    else:
        assert args.pattern == "dense", f"Invalid pattern: {args.pattern}"

    # Print time logger
    for single_block in hunyuan_video_sampler.pipeline.transformer.single_blocks:
        single_block.register_forward_hook(print_operator_log_data)
    for double_block in hunyuan_video_sampler.pipeline.transformer.double_blocks:
        double_block.register_forward_hook(print_operator_log_data)

    # Start sampling
    # TODO: batch inference check
    outputs = hunyuan_video_sampler.predict(
        prompt=args.prompt,
        height=args.video_size[0],
        width=args.video_size[1],
        video_length=args.video_length,
        seed=args.seed,
        negative_prompt=args.neg_prompt,
        infer_steps=args.infer_steps,
        guidance_scale=args.cfg_scale,
        num_videos_per_prompt=args.num_videos,
        flow_shift=args.flow_shift,
        batch_size=args.batch_size,
        embedded_guidance_scale=args.embedded_cfg_scale,
    )
    samples = outputs["samples"]

    # Save samples
    if "LOCAL_RANK" not in os.environ or int(os.environ["LOCAL_RANK"]) == 0:
        for i, sample in enumerate(samples):
            sample = samples[i].unsqueeze(0)
            save_videos_grid(sample, save_path, fps=24)
            logger.info(f"Sample save to: {save_path}")
