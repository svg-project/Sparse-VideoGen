import os

import torch

from ...logger import logger
from ..utils import visualize_sparse_bsr
from .attention import (
    WanAttn_SAPAttn_Processor,
    WanAttn_SVGAttn_Processor2_0,
    prepare_flashinfer_attention,
    prepare_flexattention,
)
from .custom_models import replace_sparse_forward
from .utils import get_attention_mask, sparsity_to_width


def replace_wan_attention(
    pipe,
    height,
    width,
    num_frames,
    first_layers_fp,
    first_times_fp,
    attention_backend="flexattn",
    pattern="SVG",  # Default to SVG for backward compatibility
    # SVG specific, but provide defaults for general call signature
    num_sampled_rows=64,
    sample_mse_max_row=10000,
    sparsity=0.25,
    # Pattern dispatcher and KMEANS_BLOCK specific args
    num_q_centroids=None,
    num_k_centroids=None,
    top_p_kmeans=None,
    min_kc_ratio=0,
    logging_file=None,
    kmeans_iter_init=0,
    kmeans_iter_step=0,
    zero_step_kmeans_init=False,
):

    context_length = 0  # This seems to be 0 for I2V in SVG
    num_frame_patches = 1 + num_frames // (pipe.vae_scale_factor_temporal * pipe.transformer.config.patch_size[0])
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    frame_patches_one_frame = int(height // mod_value) * int(width // mod_value)

    dtype = torch.bfloat16  # Or pipe.dtype
    device = pipe.device

    replace_sparse_forward()  # Assuming this is a general patch; if not, it might need to be conditional.

    num_layers = len(pipe.transformer.blocks)

    if pattern == "SVG":
        AttnModule = WanAttn_SVGAttn_Processor2_0
        AttnModule.num_sampled_rows = num_sampled_rows
        AttnModule.sample_mse_max_row = sample_mse_max_row
        AttnModule.sparsity = sparsity

        masks = ["spatial", "temporal"]
        AttnModule.attention_masks = [
            get_attention_mask(
                mask_name, sample_mse_max_row, context_length, num_frame_patches, frame_patches_one_frame
            )
            for mask_name in masks
        ]
        AttnModule.first_layers_fp = first_layers_fp
        AttnModule.first_times_fp = first_times_fp

        multiplier = diag_width = sparsity_to_width(
            sparsity, context_length, num_frame_patches, frame_patches_one_frame
        )

        AttnModule.context_length = context_length
        AttnModule.num_frame = num_frame_patches
        AttnModule.frame_size = frame_patches_one_frame

        if attention_backend == "flexattn":
            block_mask = prepare_flexattention(
                1,
                pipe.transformer.num_attention_heads,
                pipe.transformer.attention_head_dim,
                dtype,
                device,
                context_length,
                context_length,
                num_frame_patches,
                frame_patches_one_frame,
                diag_width,
                multiplier,
            )
            AttnModule.block_mask = block_mask

            logger.info("Flexattn block_mask prepared.")
            logger.info(block_mask)

        elif attention_backend == "flashinfer":
            temporal_mask_metadata = prepare_flashinfer_attention(
                1,
                pipe.transformer.num_attention_heads,
                pipe.transformer.attention_head_dim,
                dtype,
                device,
                context_length,
                context_length,
                num_frame_patches,
                frame_patches_one_frame,
                diag_width,
                multiplier,
            )
            AttnModule.temporal_mask_metadata = temporal_mask_metadata

            print(
                visualize_sparse_bsr(
                    temporal_mask_metadata[0],
                    temporal_mask_metadata[1],
                    temporal_mask_metadata[2],
                )
            )

            logger.info("Flashinfer temporal_mask_metadata prepared.")
        else:
            raise ValueError(f"Attention backend {attention_backend} not supported")

        for layer_idx, m in enumerate(pipe.transformer.blocks):
            if hasattr(m.attn1, "processor"):  # Check if processor exists
                # Ensure layer_idx is set for SVG processor logic if it relies on it being an instance property after init
                current_processor = AttnModule(layer_idx=layer_idx)  # Instantiate with layer_idx
                current_processor.num_layers = num_layers
                # Other SVG specific properties already set on AttnModule class can be used or copied if needed
                m.attn1.set_processor(current_processor)

    elif pattern == "SAP":

        # Pass K-means specific parameters to the processor's constructor or set them as attributes
        # The processor itself will handle the K-means logic internally
        logger.info(
            f"Configuring KMEANS_BLOCK attention with QC: {num_q_centroids}, KC: {num_k_centroids}, P: {top_p_kmeans}, min_kc_ratio: {min_kc_ratio}"
        )

        # Make dir and clear the logging file
        if logging_file is not None:
            os.makedirs(os.path.dirname(logging_file), exist_ok=True)
            with open(logging_file, "w") as f:
                f.write("")

        AttnModule = WanAttn_SAPAttn_Processor

        AttnModule.first_layers_fp = first_layers_fp
        AttnModule.first_times_fp = first_times_fp
        AttnModule.logging_file = logging_file

        # These might be needed by the processor if it has to adapt to sequence dimensions
        AttnModule.context_length = context_length
        AttnModule.num_frame = num_frame_patches
        AttnModule.frame_size = frame_patches_one_frame

        AttnModule.num_q_centroids = num_q_centroids
        AttnModule.num_k_centroids = num_k_centroids
        AttnModule.top_p_kmeans = top_p_kmeans
        AttnModule.min_kc_ratio = min_kc_ratio
        AttnModule.num_layers = num_layers
        AttnModule.kmeans_iter_init = kmeans_iter_init
        AttnModule.kmeans_iter_step = kmeans_iter_step
        AttnModule.zero_step_kmeans_init = zero_step_kmeans_init

        # KMEANS_BLOCK specific params for each instance, passed at init
        # replace_sparse_forward() was called earlier, assuming it's general.

        for layer_idx, m in enumerate(pipe.transformer.blocks):
            if hasattr(m.attn1, "processor"):  # Check if processor exists
                # Instantiate KMEANS_BLOCK processor with its specific parameters
                current_processor = AttnModule(
                    layer_idx=layer_idx,
                )
                m.attn1.set_processor(current_processor)
    else:  # dense or other patterns
        raise ValueError(f"Pattern '{pattern}' not supported")

    # Common logic for processors that were set
    # The loop for m.attn1.processor.layer_idx = layer_idx was integrated into SVG specific part.
    # For KMEANS_BLOCK, layer_idx is passed during instantiation.
    # The generic loop below might be redundant if all processors handle layer_idx internally or via init.
    # Ensure Attn2 (cross-attention) is not affected if it uses a different processor type or no processor.
    # The original code iterated all Attention modules, let's refine to target only self-attention (attn1)

    print(f"Attention processors replaced with {pattern} pattern.")
