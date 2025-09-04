from typing import Any, Dict, Optional, Union

import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import FP32LayerNorm
from diffusers.models.transformers.transformer_wan import (
    WanTransformer3DModel,
    WanTransformerBlock,
)
from diffusers.utils import (
    USE_PEFT_BACKEND,
    scale_lora_layers,
    unscale_lora_layers,
)

from ...kernels.triton.layernorm import triton_layernorm_forward
from ...kernels.triton.modulate import triton_modulate_gate_residual_forward, triton_modulate_shift_forward
from ...logger import logger
from ...timer import time_logging_decorator
from .attention import ENABLE_FAST_KERNEL


class WanTransformerBlock_Sparse(WanTransformerBlock):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        timestep: int = 0,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table + temb.float()
        ).chunk(6, dim=1)

        # 1. Self-attention
        with time_logging_decorator("Level 1 - layernorm"):
            if isinstance(self.norm1, FP32LayerNorm):
                if ENABLE_FAST_KERNEL:
                    norm_hidden_states = triton_layernorm_forward(
                        hidden_states, self.norm1.weight, self.norm1.bias, self.norm1.eps, self.norm1.elementwise_affine
                    )
                else:
                    norm_hidden_states = self.norm1(hidden_states.float())
            else:
                raise ValueError(f"Unsupported norm type: {type(self.norm1)}")

        with time_logging_decorator("Level 1 - modulate"):
            if ENABLE_FAST_KERNEL:
                norm_hidden_states = triton_modulate_shift_forward(
                    norm_hidden_states, scale_msa, shift_msa, output_dtype=hidden_states.dtype
                )
            else:
                norm_hidden_states = (norm_hidden_states * (1 + scale_msa) + shift_msa).type_as(hidden_states)

        with time_logging_decorator("Level 1 - self attn"):
            attn_output = self.attn1(hidden_states=norm_hidden_states, rotary_emb=rotary_emb, timestep=timestep)
        with time_logging_decorator("Level 1 - misc"):
            if ENABLE_FAST_KERNEL:
                hidden_states = triton_modulate_gate_residual_forward(
                    hidden_states, attn_output, gate_msa, output_dtype=hidden_states.dtype
                )
            else:
                hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        with time_logging_decorator("Level 1 - layernorm"):
            if isinstance(self.norm2, FP32LayerNorm):
                if ENABLE_FAST_KERNEL:
                    norm_hidden_states = triton_layernorm_forward(
                        hidden_states, self.norm2.weight, self.norm2.bias, self.norm2.eps, self.norm2.elementwise_affine
                    ).type_as(hidden_states)
                else:
                    norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
            else:
                raise ValueError(f"Unsupported norm type: {type(self.norm2)}")

        with time_logging_decorator("Level 1 - cross attn"):
            attn_output = self.attn2(hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states)
        with time_logging_decorator("Level 1 - misc"):
            hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        with time_logging_decorator("Level 1 - layernorm"):
            if isinstance(self.norm3, FP32LayerNorm):
                if ENABLE_FAST_KERNEL:
                    norm_hidden_states = triton_layernorm_forward(
                        hidden_states, self.norm3.weight, self.norm3.bias, self.norm3.eps, self.norm3.elementwise_affine
                    )
                else:
                    norm_hidden_states = self.norm3(hidden_states.float())
            else:
                raise ValueError(f"Unsupported norm type: {type(self.norm3)}")

        with time_logging_decorator("Level 1 - modulate"):
            if ENABLE_FAST_KERNEL:
                norm_hidden_states = triton_modulate_shift_forward(
                    norm_hidden_states, c_scale_msa, c_shift_msa, output_dtype=hidden_states.dtype
                )
            else:
                norm_hidden_states = (norm_hidden_states * (1 + c_scale_msa) + c_shift_msa).type_as(hidden_states)

        with time_logging_decorator("Level 1 - ffn"):
            ff_output = self.ffn(norm_hidden_states)
        with time_logging_decorator("Level 1 - misc"):
            if ENABLE_FAST_KERNEL:
                hidden_states = triton_modulate_gate_residual_forward(
                    hidden_states, ff_output, c_gate_msa, output_dtype=hidden_states.dtype
                )
            else:
                hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

        return hidden_states


class WanTransformer3DModel_Sparse(WanTransformer3DModel):
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective.")

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.rope(hidden_states)

        if ENABLE_FAST_KERNEL:
            # Required for Sparse VideoGen Fast RoPE
            rot_real = rotary_emb.real.squeeze(0).squeeze(0).contiguous().to(torch.float32)
            rot_imag = rotary_emb.imag.squeeze(0).squeeze(0).contiguous().to(torch.float32)
            rotary_emb = (rot_real, rot_imag)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2).contiguous()

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
        else:
            for block in self.blocks:
                hidden_states = block(
                    hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, timestep=timestep
                )

        # 5. Output norm, projection & unpatchify
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)


def replace_sparse_forward():
    WanTransformerBlock.forward = WanTransformerBlock_Sparse.forward
    WanTransformer3DModel.forward = WanTransformer3DModel_Sparse.forward
