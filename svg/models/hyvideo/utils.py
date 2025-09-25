"""Mask Mod for Image2Video"""

import math
from functools import lru_cache
from math import floor

import torch
from diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video import DEFAULT_PROMPT_TEMPLATE
from torch.nn.attention.flex_attention import (
    create_block_mask,
)


@lru_cache
def create_block_mask_cached(score_mod, B, H, M, N, device="cuda", _compile=False):
    block_mask = create_block_mask(score_mod, B, H, M, N, device=device, _compile=_compile)
    return block_mask


def generate_temporal_head_mask_mod(
    context_length: int = 226, prompt_length: int = 226, num_frames: int = 13, token_per_frame: int = 1350, mul: int = 2
):

    def round_to_multiple(idx):
        return floor(idx / 128) * 128

    real_length = num_frames * token_per_frame + prompt_length

    def temporal_mask_mod(b, h, q_idx, kv_idx):
        real_mask = (kv_idx < real_length) & (q_idx < real_length)
        fake_mask = (kv_idx >= real_length) & (q_idx >= real_length)

        two_frame = round_to_multiple(mul * token_per_frame)
        temporal_head_mask = torch.abs(q_idx - kv_idx) < two_frame

        text_column_mask = (num_frames * token_per_frame <= kv_idx) & (kv_idx < real_length)
        text_row_mask = (num_frames * token_per_frame <= q_idx) & (q_idx < real_length)

        video_mask = temporal_head_mask | text_column_mask | text_row_mask
        real_mask = real_mask & video_mask

        return real_mask | fake_mask

    return temporal_mask_mod


def get_attention_mask(mask_name, sample_mse_max_row, context_length, num_frame, frame_size, device="cuda"):

    attention_mask = torch.zeros(
        (context_length + num_frame * frame_size, context_length + num_frame * frame_size), device="cpu"
    )

    # TODO: fix hard coded mask
    if mask_name == "spatial":
        pixel_attn_mask = torch.zeros_like(
            attention_mask[:-context_length, :-context_length], dtype=torch.bool, device="cpu"
        )
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
        pixel_attn_mask = torch.zeros_like(
            attention_mask[:-context_length, :-context_length], dtype=torch.bool, device=device
        )

        block_size, block_thres = 128, frame_size * 1.5
        num_block = math.ceil(num_frame * frame_size / block_size)
        for i in range(num_block):
            for j in range(num_block):
                if abs(i - j) < block_thres // block_size:
                    pixel_attn_mask[i * block_size : (i + 1) * block_size, j * block_size : (j + 1) * block_size] = 1

        pixel_attn_mask = (
            pixel_attn_mask.reshape(frame_size, num_frame, frame_size, num_frame)
            .permute(1, 0, 3, 2)
            .reshape(frame_size * num_frame, frame_size * num_frame)
        )
        attention_mask[:-context_length, :-context_length] = pixel_attn_mask

        attention_mask[-context_length:, :] = 1
        attention_mask[:, -context_length:] = 1
        # attention_mask = torch.load(f"/data/home/xihaocheng/andy_develop/tmp_data/hunyuanvideo/I2VSparse/sparseattn/v5/mask_tensor/mask_temporal.pt", map_location="cpu")
    attention_mask = attention_mask[:sample_mse_max_row].cuda()
    return attention_mask


def get_prompt_length(pipe, prompt, prompt_template=DEFAULT_PROMPT_TEMPLATE, max_sequence_length=256, device="cuda"):
    """
    Compute the prompt length for the prompt. In HunyuanVideo, we have prompt_length + unprompt_length = context_length, where context_length is a fixed value.
    We need to compute the prompt_length for the prompt in advance if using SVG to pre-compile the attention mask.
    """

    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    prompt = [prompt_template["template"].format(p) for p in prompt]

    crop_start = prompt_template.get("crop_start", None)
    if crop_start is None:
        prompt_template_input = pipe.tokenizer(
            prompt_template["template"],
            padding="max_length",
            return_tensors="pt",
            return_length=False,
            return_overflowing_tokens=False,
            return_attention_mask=False,
        )
        crop_start = prompt_template_input["input_ids"].shape[-1]
        # Remove <|eot_id|> token and placeholder {}
        crop_start -= 2

    max_sequence_length += crop_start
    text_inputs = pipe.tokenizer(
        prompt,
        max_length=max_sequence_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_length=False,
        return_overflowing_tokens=False,
        return_attention_mask=True,
    )
    text_input_ids = text_inputs.input_ids.to(device=device)
    prompt_attention_mask = text_inputs.attention_mask.to(device=device)

    if crop_start is not None and crop_start > 0:
        prompt_attention_mask = prompt_attention_mask[:, crop_start:]

    prompt_length = prompt_attention_mask.sum()
    return prompt_length


def sparsity_to_width(sparsity, context_length, num_frame, frame_size):
    seq_len = context_length + num_frame * frame_size
    total_elements = seq_len**2

    sparsity = (sparsity * total_elements - 2 * seq_len * context_length) / total_elements

    width = seq_len * (1 - math.sqrt(1 - sparsity))
    width_frame = width / frame_size

    return width_frame
