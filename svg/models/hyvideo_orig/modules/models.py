import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import ModelMixin
from einops import rearrange

from ....timer import time_logging_decorator
from .activation_layers import get_activation_layer
from .attenion import attention, get_cu_seqlens, parallel_attention
from .embed_layers import PatchEmbed, TextProjection, TimestepEmbedder
from .mlp_layers import MLP, FinalLayer, MLPEmbedder
from .modulate_layers import ModulateDiT, apply_gate, modulate
from .norm_layers import get_norm_layer
from .posemb_layers import apply_rotary_emb
from .token_refiner import SingleTokenRefiner


class MMDoubleStreamBlock(nn.Module):
    """
    A multimodal dit block with seperate modulation for
    text and image/video, see more details (SD3): https://arxiv.org/abs/2403.03206
                                     (Flux.1): https://github.com/black-forest-labs/flux
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float,
        mlp_act_type: str = "gelu_tanh",
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qkv_bias: bool = False,
        layer_idx: int = 0,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.layer_idx = layer_idx
        self.deterministic = False
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        self.img_mod = ModulateDiT(
            hidden_size,
            factor=6,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        self.img_attn_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias, **factory_kwargs)
        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.img_attn_q_norm = qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        self.img_attn_k_norm = qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        self.img_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.img_mlp = MLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            **factory_kwargs,
        )

        self.txt_mod = ModulateDiT(
            hidden_size,
            factor=6,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        self.txt_attn_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias, **factory_kwargs)
        self.txt_attn_q_norm = qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        self.txt_attn_k_norm = qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        self.txt_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.txt_mlp = MLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            **factory_kwargs,
        )
        self.hybrid_seq_parallel_attn = None

        # Sparsity
        self.sparse_args = None

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

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

        with time_logging_decorator("Level 1 Double - Modulation"):
            (
                img_mod1_shift,
                img_mod1_scale,
                img_mod1_gate,
                img_mod2_shift,
                img_mod2_scale,
                img_mod2_gate,
            ) = self.img_mod(
                vec
            ).chunk(6, dim=-1)
            (
                txt_mod1_shift,
                txt_mod1_scale,
                txt_mod1_gate,
                txt_mod2_shift,
                txt_mod2_scale,
                txt_mod2_gate,
            ) = self.txt_mod(
                vec
            ).chunk(6, dim=-1)

            # Prepare image for attention.
            img_modulated = self.img_norm1(img)
            img_modulated = modulate(img_modulated, shift=img_mod1_shift, scale=img_mod1_scale)

        with time_logging_decorator("Level 1 Double - Image Linear QKV"):
            img_qkv = self.img_attn_qkv(img_modulated)
            img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)

        with time_logging_decorator("Level 1 Double - QK-Norm"):
            # Apply QK-Norm if needed
            img_q = self.img_attn_q_norm(img_q).to(img_v)
            img_k = self.img_attn_k_norm(img_k).to(img_v)

        with time_logging_decorator("Level 1 Double - RoPE"):
            # Apply RoPE if needed.
            if freqs_cis is not None:
                img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
                assert img_qq.shape == img_q.shape and img_kk.shape == img_k.shape, f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
                img_q, img_k = img_qq, img_kk

        with time_logging_decorator("Level 1 Double - Text Misc Bgn"):
            # Prepare txt for attention.
            txt_modulated = self.txt_norm1(txt)
            txt_modulated = modulate(txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale)
            txt_qkv = self.txt_attn_qkv(txt_modulated)
            txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)
            # Apply QK-Norm if needed.
            txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
            txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)

        with time_logging_decorator("Level 1 Double - Image and Text Concat"):
            # Run actual attention.
            q = torch.cat((img_q, txt_q), dim=1)
            k = torch.cat((img_k, txt_k), dim=1)
            v = torch.cat((img_v, txt_v), dim=1)
            assert cu_seqlens_q.shape[0] == 2 * img.shape[0] + 1, f"cu_seqlens_q.shape:{cu_seqlens_q.shape}, img.shape[0]:{img.shape[0]}"

        with time_logging_decorator("Level 1 Double - Attention"):
            # attention computation start
            if not self.hybrid_seq_parallel_attn:
                attn = attention(
                    q,
                    k,
                    v,
                    mode="flash",
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_kv=cu_seqlens_kv,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_kv=max_seqlen_kv,
                    batch_size=img_k.shape[0],
                    timestep=timestep,
                    layer_idx=self.layer_idx,
                )
            else:
                attn = parallel_attention(self.hybrid_seq_parallel_attn, q, k, v, img_q_len=img_q.shape[1], img_kv_len=img_k.shape[1], cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv)

            # attention computation end

            img_attn, txt_attn = attn[:, : img.shape[1]], attn[:, img.shape[1] :]

        with time_logging_decorator("Level 1 Double - Image Linear AttnO"):
            # Calculate the img bloks.
            img_attn = self.img_attn_proj(img_attn)
            img_attn = apply_gate(img_attn, gate=img_mod1_gate)
            img = img + img_attn

        with time_logging_decorator("Level 1 Double - Image MLP"):
            img_mlp = modulate(self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale)
            img_mlp = self.img_mlp(img_mlp)
            img_mlp = apply_gate(img_mlp, gate=img_mod2_gate)
            img = img + img_mlp

        with time_logging_decorator("Level 1 Double - Text Misc End"):
            # Calculate the txt bloks.
            txt_attn = self.txt_attn_proj(txt_attn)
            txt_attn = apply_gate(txt_attn, gate=txt_mod1_gate)
            txt = txt + txt_attn

            txt_mlp = modulate(self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale)
            txt_mlp = self.txt_mlp(txt_mlp)
            txt_mlp = apply_gate(txt_mlp, gate=txt_mod2_gate)
            txt = txt + txt_mlp

        return img, txt


class MMSingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    Also refer to (SD3): https://arxiv.org/abs/2403.03206
                  (Flux.1): https://github.com/black-forest-labs/flux
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qk_scale: float = None,
        layer_idx: int = 0,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.layer_idx = layer_idx
        self.deterministic = False
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)
        self.mlp_hidden_dim = mlp_hidden_dim
        self.scale = qk_scale or head_dim**-0.5

        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + mlp_hidden_dim, **factory_kwargs)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + mlp_hidden_dim, hidden_size, **factory_kwargs)

        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.q_norm = qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        self.k_norm = qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()

        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        self.mlp_act = get_activation_layer(mlp_act_type)()
        self.modulation = ModulateDiT(
            hidden_size,
            factor=3,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.hybrid_seq_parallel_attn = None

        # Sparsity
        self.sparse_args = None

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

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

        with time_logging_decorator("Level 1 Single - Modulation"):
            mod_shift, mod_scale, mod_gate = self.modulation(vec).chunk(3, dim=-1)
            x_mod = modulate(self.pre_norm(x), shift=mod_shift, scale=mod_scale)

        with time_logging_decorator("Level 1 Single - Linear QKV"):
            qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

            q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)

            # Cause Inefficiency
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()

        with time_logging_decorator("Level 1 Single - QK-Norm"):
            # Apply QK-Norm if needed.
            q = self.q_norm(q).to(v)
            k = self.k_norm(k).to(v)

        with time_logging_decorator("Level 1 Single - RoPE"):
            # Apply RoPE if needed.
            if freqs_cis is not None:
                img_q, txt_q = q[:, :, :-txt_len, :], q[:, :, -txt_len:, :]
                img_k, txt_k = k[:, :, :-txt_len, :], k[:, :, -txt_len:, :]

                img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=True)
                assert img_qq.shape == img_q.shape and img_kk.shape == img_k.shape, f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
                img_q, img_k = img_qq, img_kk

                q = torch.cat((img_q, txt_q), dim=2)
                k = torch.cat((img_k, txt_k), dim=2)

                # Cause Inefficiency
                q = q.transpose(1, 2).contiguous()
                k = k.transpose(1, 2).contiguous()

            # Compute attention.
            assert cu_seqlens_q.shape[0] == 2 * x.shape[0] + 1, f"cu_seqlens_q.shape:{cu_seqlens_q.shape}, x.shape[0]:{x.shape[0]}"

        with time_logging_decorator("Level 1 Single - Attention"):
            # attention computation start
            if not self.hybrid_seq_parallel_attn:
                attn = attention(
                    q,
                    k,
                    v,
                    mode="flash",
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_kv=cu_seqlens_kv,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_kv=max_seqlen_kv,
                    batch_size=x.shape[0],
                    timestep=timestep,
                    layer_idx=self.layer_idx,
                )
            else:
                attn = parallel_attention(self.hybrid_seq_parallel_attn, q, k, v, img_q_len=img_q.shape[1], img_kv_len=img_k.shape[1], cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv)
            # attention computation end

        with time_logging_decorator("Level 1 Single - MLP"):
            # Compute activation in mlp stream, cat again and run second linear layer.
            output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
            output = apply_gate(output, gate=mod_gate)
            x = x + output

        return x


class HYVideoDiffusionTransformer(ModelMixin, ConfigMixin):
    """
    HunyuanVideo Transformer backbone

    Inherited from ModelMixin and ConfigMixin for compatibility with diffusers' sampler StableDiffusionPipeline.

    Reference:
    [1] Flux.1: https://github.com/black-forest-labs/flux
    [2] MMDiT: http://arxiv.org/abs/2403.03206

    Parameters
    ----------
    args: argparse.Namespace
        The arguments parsed by argparse.
    patch_size: list
        The size of the patch.
    in_channels: int
        The number of input channels.
    out_channels: int
        The number of output channels.
    hidden_size: int
        The hidden size of the transformer backbone.
    heads_num: int
        The number of attention heads.
    mlp_width_ratio: float
        The ratio of the hidden size of the MLP in the transformer block.
    mlp_act_type: str
        The activation function of the MLP in the transformer block.
    depth_double_blocks: int
        The number of transformer blocks in the double blocks.
    depth_single_blocks: int
        The number of transformer blocks in the single blocks.
    rope_dim_list: list
        The dimension of the rotary embedding for t, h, w.
    qkv_bias: bool
        Whether to use bias in the qkv linear layer.
    qk_norm: bool
        Whether to use qk norm.
    qk_norm_type: str
        The type of qk norm.
    guidance_embed: bool
        Whether to use guidance embedding for distillation.
    text_projection: str
        The type of the text projection, default is single_refiner.
    use_attention_mask: bool
        Whether to use attention mask for text encoder.
    dtype: torch.dtype
        The dtype of the model.
    device: torch.device
        The device of the model.
    """

    @register_to_config
    def __init__(
        self,
        args: Any,
        patch_size: list = [1, 2, 2],
        in_channels: int = 4,  # Should be VAE.config.latent_channels.
        out_channels: int = None,
        hidden_size: int = 3072,
        heads_num: int = 24,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        mm_double_blocks_depth: int = 20,
        mm_single_blocks_depth: int = 40,
        rope_dim_list: List[int] = [16, 56, 56],
        qkv_bias: bool = True,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        guidance_embed: bool = False,  # For modulation.
        text_projection: str = "single_refiner",
        use_attention_mask: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.unpatchify_channels = self.out_channels
        self.guidance_embed = guidance_embed
        self.rope_dim_list = rope_dim_list

        # Text projection. Default to linear projection.
        # Alternative: TokenRefiner. See more details (LI-DiT): http://arxiv.org/abs/2406.11831
        self.use_attention_mask = use_attention_mask
        self.text_projection = text_projection

        self.text_states_dim = args.text_states_dim
        self.text_states_dim_2 = args.text_states_dim_2

        if hidden_size % heads_num != 0:
            raise ValueError(f"Hidden size {hidden_size} must be divisible by heads_num {heads_num}")
        pe_dim = hidden_size // heads_num
        if sum(rope_dim_list) != pe_dim:
            raise ValueError(f"Got {rope_dim_list} but expected positional dim {pe_dim}")
        self.hidden_size = hidden_size
        self.heads_num = heads_num

        # image projection
        self.img_in = PatchEmbed(self.patch_size, self.in_channels, self.hidden_size, **factory_kwargs)

        # text projection
        if self.text_projection == "linear":
            self.txt_in = TextProjection(
                self.text_states_dim,
                self.hidden_size,
                get_activation_layer("silu"),
                **factory_kwargs,
            )
        elif self.text_projection == "single_refiner":
            self.txt_in = SingleTokenRefiner(self.text_states_dim, hidden_size, heads_num, depth=2, **factory_kwargs)
        else:
            raise NotImplementedError(f"Unsupported text_projection: {self.text_projection}")

        # time modulation
        self.time_in = TimestepEmbedder(self.hidden_size, get_activation_layer("silu"), **factory_kwargs)

        # text modulation
        self.vector_in = MLPEmbedder(self.text_states_dim_2, self.hidden_size, **factory_kwargs)

        # guidance modulation
        self.guidance_in = TimestepEmbedder(self.hidden_size, get_activation_layer("silu"), **factory_kwargs) if guidance_embed else None

        # double blocks
        self.double_blocks = nn.ModuleList(
            [
                MMDoubleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    qkv_bias=qkv_bias,
                    layer_idx=layer_idx,
                    **factory_kwargs,
                )
                for layer_idx in range(mm_double_blocks_depth)
            ]
        )

        # single blocks
        self.single_blocks = nn.ModuleList(
            [
                MMSingleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    layer_idx=layer_idx,
                    **factory_kwargs,
                )
                for layer_idx in range(mm_double_blocks_depth, mm_double_blocks_depth + mm_single_blocks_depth)
            ]
        )

        self.final_layer = FinalLayer(
            self.hidden_size,
            self.patch_size,
            self.out_channels,
            get_activation_layer("silu"),
            **factory_kwargs,
        )

    def enable_deterministic(self):
        for block in self.double_blocks:
            block.enable_deterministic()
        for block in self.single_blocks:
            block.enable_deterministic()

    def disable_deterministic(self):
        for block in self.double_blocks:
            block.disable_deterministic()
        for block in self.single_blocks:
            block.disable_deterministic()

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,  # Should be in range(0, 1000).
        text_states: torch.Tensor = None,
        text_mask: torch.Tensor = None,  # Now we don't use it.
        text_states_2: Optional[torch.Tensor] = None,  # Text embedding for modulation.
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        guidance: torch.Tensor = None,  # Guidance for modulation, should be cfg_scale x 1000.
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        with time_logging_decorator("Level 0 - Before Main Blocks"):
            out = {}
            img = x
            txt = text_states
            _, _, ot, oh, ow = x.shape
            tt, th, tw = (
                ot // self.patch_size[0],
                oh // self.patch_size[1],
                ow // self.patch_size[2],
            )

            # Prepare modulation vectors.
            vec = self.time_in(t)

            # text modulation
            vec = vec + self.vector_in(text_states_2)

            # guidance modulation
            if self.guidance_embed:
                if guidance is None:
                    raise ValueError("Didn't get guidance strength for guidance distilled model.")

                # our timestep_embedding is merged into guidance_in(TimestepEmbedder)
                vec = vec + self.guidance_in(guidance)

            # Embed image and text.
            img = self.img_in(img)
            if self.text_projection == "linear":
                txt = self.txt_in(txt)
            elif self.text_projection == "single_refiner":
                txt = self.txt_in(txt, t, text_mask if self.use_attention_mask else None)
            else:
                raise NotImplementedError(f"Unsupported text_projection: {self.text_projection}")

            txt_seq_len = txt.shape[1]
            img_seq_len = img.shape[1]

            # Compute cu_squlens and max_seqlen for flash attention
            cu_seqlens_q = get_cu_seqlens(text_mask, img_seq_len)
            cu_seqlens_kv = cu_seqlens_q
            max_seqlen_q = img_seq_len + txt_seq_len
            max_seqlen_kv = max_seqlen_q

            if hasattr(self, "sparse_args"):
                if getattr(self.sparse_args, "pattern", None) in ["SVG", "SAP"]:
                    freqs_cos = freqs_cos.to(x.device).to(torch.float32)
                    freqs_sin = freqs_sin.to(x.device).to(torch.float32)
            freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None

        with time_logging_decorator("Level 0 - Double Blocks"):
            # --------------------- Pass through DiT blocks ------------------------
            for _, block in enumerate(self.double_blocks):
                double_block_args = [img, txt, vec, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, freqs_cis, t]

                img, txt = block(*double_block_args)

        with time_logging_decorator("Level 0 - Single Blocks"):
            # Merge txt and img to pass through single stream blocks.
            x = torch.cat((img, txt), 1)
            if len(self.single_blocks) > 0:
                for _, block in enumerate(self.single_blocks):
                    single_block_args = [x, vec, txt_seq_len, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, (freqs_cos, freqs_sin), t]

                    x = block(*single_block_args)

        with time_logging_decorator("Level 0 - After Main Blocks"):
            img = x[:, :img_seq_len, ...]

            # ---------------------------- Final layer ------------------------------
            img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)

            img = self.unpatchify(img, tt, th, tw)
            if return_dict:
                out["x"] = img
                return out
        return img

    def unpatchify(self, x, t, h, w):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.unpatchify_channels
        pt, ph, pw = self.patch_size
        assert t * h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], t, h, w, c, pt, ph, pw))
        x = torch.einsum("nthwcopq->nctohpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))

        return imgs

    def params_count(self):
        counts = {
            "double": sum(
                [
                    sum(p.numel() for p in block.img_attn_qkv.parameters())
                    + sum(p.numel() for p in block.img_attn_proj.parameters())
                    + sum(p.numel() for p in block.img_mlp.parameters())
                    + sum(p.numel() for p in block.txt_attn_qkv.parameters())
                    + sum(p.numel() for p in block.txt_attn_proj.parameters())
                    + sum(p.numel() for p in block.txt_mlp.parameters())
                    for block in self.double_blocks
                ]
            ),
            "single": sum([sum(p.numel() for p in block.linear1.parameters()) + sum(p.numel() for p in block.linear2.parameters()) for block in self.single_blocks]),
            "total": sum(p.numel() for p in self.parameters()),
        }
        counts["attn+mlp"] = counts["double"] + counts["single"]
        return counts


#################################################################################
#                             HunyuanVideo Configs                              #
#################################################################################

HUNYUAN_VIDEO_CONFIG = {
    "HYVideo-T/2": {
        "mm_double_blocks_depth": 20,
        "mm_single_blocks_depth": 40,
        "rope_dim_list": [16, 56, 56],
        "hidden_size": 3072,
        "heads_num": 24,
        "mlp_width_ratio": 4,
    },
    "HYVideo-T/2-cfgdistill": {
        "mm_double_blocks_depth": 20,
        "mm_single_blocks_depth": 40,
        "rope_dim_list": [16, 56, 56],
        "hidden_size": 3072,
        "heads_num": 24,
        "mlp_width_ratio": 4,
        "guidance_embed": True,
    },
}
