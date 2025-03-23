import torch
import torch.nn.functional as F

from diffusers.models.attention_processor import Attention
from diffusers.utils import export_to_video, load_image

from .attention import WanAttn_SparseAttn_Processor2_0, prepare_flexattention
from .utils import sparsity_to_width, get_attention_mask
from .custom_models import replace_sparse_forward

def replace_wan_attention(pipe, num_sampled_rows, sample_mse_max_row, sparsity, first_layers_fp, first_times_fp):

    masks = ["spatial", "temporal"]

    context_length = 0
    num_frame = 21
    frame_size = 3600
    
    dtype = torch.bfloat16

    AttnModule = WanAttn_SparseAttn_Processor2_0
    AttnModule.num_sampled_rows = num_sampled_rows
    AttnModule.sample_mse_max_row = sample_mse_max_row
    AttnModule.attention_masks = [get_attention_mask(mask_name, sample_mse_max_row, context_length, num_frame, frame_size) for mask_name in masks]
    AttnModule.first_layers_fp = first_layers_fp
    AttnModule.first_times_fp = first_times_fp

    multiplier = diag_width = sparsity_to_width(sparsity, context_length, num_frame, frame_size)

    AttnModule.context_length = context_length
    AttnModule.num_frame = num_frame
    AttnModule.frame_size = frame_size
    
    # NOTE: ??? Prepare placement will strongly decrease PSNR
    # prepare_placement(2, 48, 64, dtype, "cuda", context_length, num_frame, frame_size)
    block_mask = prepare_flexattention(1, 40, 128, dtype, "cuda", context_length, context_length, num_frame, frame_size, diag_width, multiplier)
    AttnModule.block_mask = block_mask
    
    print(block_mask)
    
    replace_sparse_forward()
    
    num_layers = len(pipe.transformer.blocks)

    for layer_idx, m in enumerate(pipe.transformer.blocks):
        m.attn1.processor.layer_idx = layer_idx
        
    for _ , m in pipe.transformer.named_modules():
        if isinstance(m, Attention):
            if hasattr(m.processor, "layer_idx"): # Only Attn 1, No Attn 2
                layer_idx = m.processor.layer_idx
                m.set_processor(AttnModule(layer_idx))
                m.processor.num_layers = num_layers