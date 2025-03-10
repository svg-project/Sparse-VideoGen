import os
import time
import math
import json
from pathlib import Path
from loguru import logger
from datetime import datetime

import torch
from svg.models.hyvideo.utils.file_utils import save_videos_grid
from svg.models.hyvideo.config import parse_args
from svg.models.hyvideo.inference import HunyuanVideoSampler


def sparsity_to_width(sparsity, context_length, num_frame, frame_size):
    seq_len = context_length + num_frame * frame_size
    total_elements = seq_len ** 2
    
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
        embedded_guidance_scale=args.embedded_cfg_scale
    )
    prompt_len = prompt_mask.sum()

    print(f"Memory: {torch.cuda.memory_allocated() // 1024 ** 2} / {torch.cuda.max_memory_allocated() // 1024 ** 2} MB before Inference")

    cfg_size, num_head, head_dim, dtype, device = 1, 24, 128, torch.bfloat16, "cuda"
    context_length, num_frame, frame_size = 256, 33, 3600

    # Calculation
    spatial_width = temporal_width = sparsity_to_width(args.sparsity, context_length, num_frame, frame_size)
                
    print(f"Spatial_width: {spatial_width}, Temporal_width: {temporal_width}. Sparsity: {args.sparsity}")

    save_path = args.output_path
    if not os.path.exists(args.save_path):
        os.makedirs(save_path, exist_ok=True)
        
    if args.pattern == "SVG":
        masks = ["spatial", "temporal"]

        def get_attention_mask(version, mask_name):
            print(f"Loading Attention Sparse Mask {mask_name}")
            print(f"Memory: {torch.cuda.memory_allocated() // 1024 ** 2} / {torch.cuda.max_memory_allocated() // 1024 ** 2} MB before Loading Mask")

            attention_type = mask_name.split("sparse")[-1]
            attention_mask = torch.load(f"sparseattn/{version}/mask_tensor/mask_{attention_type}.pt", map_location="cpu")
            attention_mask = attention_mask[:args.sample_mse_max_row].cuda()
        
            print(f"Memory: {torch.cuda.memory_allocated() // 1024 ** 2} / {torch.cuda.max_memory_allocated() // 1024 ** 2} MB after Loading Mask")

            return attention_mask


        from svg.models.hyvideo.modules.attenion import Hunyuan_SparseAttn, prepare_flexattention
        from svg.models.hyvideo.modules.custom_models import replace_sparse_forward

        AttnModule = Hunyuan_SparseAttn
        AttnModule.num_sampled_rows = args.num_sampled_rows
        AttnModule.sample_mse_max_row = args.sample_mse_max_row
        AttnModule.attention_masks = [get_attention_mask(args.version, mask_name) for mask_name in masks]
        AttnModule.version = args.version

        block_mask = prepare_flexattention(
                cfg_size, num_head, head_dim, dtype, device, 
                context_length, prompt_len, num_frame, frame_size, 
                diag_width=spatial_width, multiplier=temporal_width
            )
        AttnModule.block_mask = block_mask
        replace_sparse_forward()


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
        embedded_guidance_scale=args.embedded_cfg_scale
    )
    samples = outputs['samples']
    
    # Save samples
    if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
        for i, sample in enumerate(samples):
            sample = samples[i].unsqueeze(0)
            save_videos_grid(sample, save_path, fps=24)
            logger.info(f'Sample save to: {save_path}')

