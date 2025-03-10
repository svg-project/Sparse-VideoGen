import os
from typing import Any, List, Tuple, Optional, Union, Dict
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

from .activation_layers import get_activation_layer
from .norm_layers import get_norm_layer
from .embed_layers import TimestepEmbedder, PatchEmbed, TextProjection
from .attenion import attention, parallel_attention, get_cu_seqlens
from .posemb_layers import apply_rotary_emb
from .mlp_layers import MLP, MLPEmbedder, FinalLayer
from .modulate_layers import ModulateDiT, modulate, apply_gate
from .token_refiner import SingleTokenRefiner
from .models import MMDoubleStreamBlock, MMSingleStreamBlock
# import sys
# sys.path.append('kernels/build/')
# import _kernels


class MMDoubleStreamBlock_Sparse(MMDoubleStreamBlock):
    """
    A multimodal dit block with seperate modulation for
    text and image/video, see more details (SD3): https://arxiv.org/abs/2403.03206
                                     (Flux.1): https://github.com/black-forest-labs/flux
    """

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        freqs_cis: tuple = None,
        timestep: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        (
            img_mod1_shift,
            img_mod1_scale,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
        ) = self.img_mod(vec).chunk(6, dim=-1)
        (
            txt_mod1_shift,
            txt_mod1_scale,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate,
        ) = self.txt_mod(vec).chunk(6, dim=-1)

        # Prepare image for attention.
        img_modulated = self.img_norm1(img)
        img_modulated = modulate(
            img_modulated, shift=img_mod1_shift, scale=img_mod1_scale
        )

        img_qkv = self.img_attn_qkv(img_modulated)

        img_q, img_k, img_v = rearrange(
            img_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
        )

        # # Cause Inefficiency
        # img_q = img_q.transpose(1, 2).contiguous()
        # img_k = img_k.transpose(1, 2).contiguous()
        
        # # Apply QK-Norm if needed
        # _kernels.rms_norm_forward(img_q.view(-1, img_q.shape[-1]), self.img_attn_q_norm.weight, self.img_attn_q_norm.eps)
        # _kernels.rms_norm_forward(img_k.view(-1, img_k.shape[-1]), self.img_attn_k_norm.weight, self.img_attn_k_norm.eps)

        # # Apply RoPE if needed.
        # if freqs_cis is not None:
        #     cos, sin = freqs_cis[0], freqs_cis[1]
        #     _kernels.apply_qk_rope_inplace_cossin_txtlast(img_q, img_k, cos, sin, 0)

        # # Cause Inefficiency
        # img_q = img_q.transpose(1, 2).contiguous()
        # img_k = img_k.transpose(1, 2).contiguous()

        # Apply QK-Norm if needed
        img_q = self.img_attn_q_norm(img_q).to(img_v)
        img_k = self.img_attn_k_norm(img_k).to(img_v)

        # Apply RoPE if needed.
        if freqs_cis is not None:
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            assert (
                img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
            ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            img_q, img_k = img_qq, img_kk
        
        # Prepare txt for attention.
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = modulate(
            txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale
        )
        txt_qkv = self.txt_attn_qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(
            txt_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
        )

        # Apply QK-Norm if needed.
        txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
        txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)

        # Run actual attention.
        q = torch.cat((img_q, txt_q), dim=1)
        k = torch.cat((img_k, txt_k), dim=1)
        v = torch.cat((img_v, txt_v), dim=1)
        assert (
            cu_seqlens_q.shape[0] == 2 * img.shape[0] + 1
        ), f"cu_seqlens_q.shape:{cu_seqlens_q.shape}, img.shape[0]:{img.shape[0]}"

        # attention computation start
        if not self.hybrid_seq_parallel_attn:
            attn = attention(
                q,
                k,
                v,
                mode="sparse",
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                batch_size=img_k.shape[0],
                timestep=timestep,
                layer_idx=self.layer_idx
            )
        else:
            attn = parallel_attention(
                self.hybrid_seq_parallel_attn,
                q,
                k,
                v,
                img_q_len=img_q.shape[1],
                img_kv_len=img_k.shape[1],
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv
            )
    
        # attention computation end

        img_attn, txt_attn = attn[:, : img.shape[1]], attn[:, img.shape[1] :]

        # Calculate the img bloks.
        img = img + apply_gate(self.img_attn_proj(img_attn), gate=img_mod1_gate)
        img = img + apply_gate(
            self.img_mlp(
                modulate(
                    self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale
                )
            ),
            gate=img_mod2_gate,
        )

        # Calculate the txt bloks.
        txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate)
        txt = txt + apply_gate(
            self.txt_mlp(
                modulate(
                    self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale
                )
            ),
            gate=txt_mod2_gate,
        )

        return img, txt


class MMSingleStreamBlock_Sparse(MMSingleStreamBlock):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    Also refer to (SD3): https://arxiv.org/abs/2403.03206
                  (Flux.1): https://github.com/black-forest-labs/flux
    """

    def forward(
        self,
        x: torch.Tensor,
        vec: torch.Tensor,
        txt_len: int,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
        timestep: torch.Tensor = None,
    ) -> torch.Tensor:
        mod_shift, mod_scale, mod_gate = self.modulation(vec).chunk(3, dim=-1)
        x_mod = modulate(self.pre_norm(x), shift=mod_shift, scale=mod_scale)
        qkv, mlp = torch.split(
            self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1
        )

        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)

        # # Cause Inefficiency
        # q = q.transpose(1, 2).contiguous()
        # k = k.transpose(1, 2).contiguous()
        
        # # Apply QK-Norm if needed.
        # _kernels.rms_norm_forward(q.view(-1, q.shape[-1]), self.q_norm.weight, self.q_norm.eps)
        # _kernels.rms_norm_forward(k.view(-1, k.shape[-1]), self.k_norm.weight, self.k_norm.eps)

        # # Apply RoPE if needed.
        # if freqs_cis is not None:
        #     cos, sin = freqs_cis[0], freqs_cis[1]
        #     _kernels.apply_qk_rope_inplace_cossin_txtlast(q, k, cos, sin, txt_len)

        #     # Cause Inefficiency
        #     q = q.transpose(1, 2).contiguous()
        #     k = k.transpose(1, 2).contiguous()

        # Cause Inefficiency
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        
        # Apply QK-Norm if needed.
        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)

        # Apply RoPE if needed.
        if freqs_cis is not None:
            img_q, txt_q = q[:, :, :-txt_len, :], q[:, :, -txt_len:, :]
            img_k, txt_k = k[:, :, :-txt_len, :], k[:, :, -txt_len:, :]

            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=True)
            assert (
                img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
            ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            img_q, img_k = img_qq, img_kk

            q = torch.cat((img_q, txt_q), dim=2)
            k = torch.cat((img_k, txt_k), dim=2)

            # Cause Inefficiency
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
        
        # Compute attention.
        assert (
            cu_seqlens_q.shape[0] == 2 * x.shape[0] + 1
        ), f"cu_seqlens_q.shape:{cu_seqlens_q.shape}, x.shape[0]:{x.shape[0]}"

        # attention computation start
        attn = attention(
            q,
            k,
            v,
            mode="sparse",
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            batch_size=x.shape[0],
            timestep=timestep,
            layer_idx=self.layer_idx
        )
        # attention computation end

        # Compute activation in mlp stream, cat again and run second linear layer.
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))

        return x + apply_gate(output, gate=mod_gate)


def replace_sparse_forward():
    MMDoubleStreamBlock.forward = MMDoubleStreamBlock_Sparse.forward
    MMSingleStreamBlock.forward = MMSingleStreamBlock_Sparse.forward

