import torch
import torch.nn as nn
from typing import Optional, Tuple
from diffusers.models.attention_processor import Attention
from svg.timer import time_logging_decorator

from diffusers.models.transformers.transformer_cosmos import CosmosTransformer3DModel, CosmosTransformerBlock, Transformer2DModelOutput
from diffusers.utils import is_torchvision_available

if is_torchvision_available():
    from torchvision import transforms

class CosmosTransformerBlock_Sparse(CosmosTransformerBlock):
    @time_logging_decorator("Level 0 - forward")
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        embedded_timestep: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        extra_pos_emb: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if extra_pos_emb is not None:
            hidden_states = hidden_states + extra_pos_emb

        # 1. Self Attention
        with time_logging_decorator("Level 1 - Norm 1"):
            norm_hidden_states, gate = self.norm1(hidden_states, embedded_timestep, temb)
        with time_logging_decorator("Level 1 - Attn1"):
            attn_output = self.attn1(norm_hidden_states, image_rotary_emb=image_rotary_emb, timestep=timestep)
        with time_logging_decorator("Level 1 - Add"):
            hidden_states = hidden_states + gate * attn_output

        # 2. Cross Attention
        with time_logging_decorator("Level 1 - Norm 2"):
            norm_hidden_states, gate = self.norm2(hidden_states, embedded_timestep, temb)
        with time_logging_decorator("Level 1 - Attn2"):
            attn_output = self.attn2(
                norm_hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
            )
        with time_logging_decorator("Level 1 - Add"):
            hidden_states = hidden_states + gate * attn_output

        # 3. Feed Forward
        with time_logging_decorator("Level 1 - Norm 3"):
            norm_hidden_states, gate = self.norm3(hidden_states, embedded_timestep, temb)
        with time_logging_decorator("Level 1 - FF"):
            ff_output = self.ff(norm_hidden_states)
        with time_logging_decorator("Level 1 - Add"):
            hidden_states = hidden_states + gate * ff_output

        return hidden_states



class CosmosTransformer3DModel_Sparse(CosmosTransformer3DModel):
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        fps: Optional[int] = None,
        condition_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape

        # 1. Concatenate padding mask if needed & prepare attention mask
        if condition_mask is not None:
            hidden_states = torch.cat([hidden_states, condition_mask], dim=1)

        if self.config.concat_padding_mask:
            padding_mask = transforms.functional.resize(
                padding_mask, list(hidden_states.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )
            hidden_states = torch.cat(
                [hidden_states, padding_mask.unsqueeze(2).repeat(batch_size, 1, num_frames, 1, 1)], dim=1
            )

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, S]

        # 2. Generate positional embeddings
        image_rotary_emb = self.rope(hidden_states, fps=fps)
        extra_pos_emb = self.learnable_pos_embed(hidden_states) if self.config.extra_pos_embed_type else None

        # 3. Patchify input
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w
        hidden_states = self.patch_embed(hidden_states)
        hidden_states = hidden_states.flatten(1, 3)  # [B, T, H, W, C] -> [B, THW, C]

        # 4. Timestep embeddings
        if timestep.ndim == 1:
            temb, embedded_timestep = self.time_embed(hidden_states, timestep)
        elif timestep.ndim == 5:
            assert timestep.shape == (batch_size, 1, num_frames, 1, 1), (
                f"Expected timestep to have shape [B, 1, T, 1, 1], but got {timestep.shape}"
            )
            timestep = timestep.flatten()
            temb, embedded_timestep = self.time_embed(hidden_states, timestep)
            # We can do this because num_frames == post_patch_num_frames, as p_t is 1
            temb, embedded_timestep = (
                x.view(batch_size, post_patch_num_frames, 1, 1, -1)
                .expand(-1, -1, post_patch_height, post_patch_width, -1)
                .flatten(1, 3)
                for x in (temb, embedded_timestep)
            )  # [BT, C] -> [B, T, 1, 1, C] -> [B, T, H, W, C] -> [B, THW, C]
        else:
            assert False

        # 5. Transformer blocks
        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    embedded_timestep,
                    temb,
                    image_rotary_emb,
                    extra_pos_emb,
                    attention_mask,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    embedded_timestep=embedded_timestep,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    timestep=timestep,
                    extra_pos_emb=extra_pos_emb,
                    attention_mask=attention_mask,
                )

        # 6. Output norm & projection & unpatchify
        hidden_states = self.norm_out(hidden_states, embedded_timestep, temb)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.unflatten(2, (p_h, p_w, p_t, -1))
        hidden_states = hidden_states.unflatten(1, (post_patch_num_frames, post_patch_height, post_patch_width))
        # NOTE: The permutation order here is not the inverse operation of what happens when patching as usually expected.
        # It might be a source of confusion to the reader, but this is correct
        hidden_states = hidden_states.permute(0, 7, 1, 6, 2, 4, 3, 5)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (hidden_states,)

        return Transformer2DModelOutput(sample=hidden_states)


def replace_sparse_forward():
    CosmosTransformerBlock.forward = CosmosTransformerBlock_Sparse.forward
    CosmosTransformer3DModel.forward = CosmosTransformer3DModel_Sparse.forward