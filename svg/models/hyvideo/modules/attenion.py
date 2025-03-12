import math

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention

from flash_attn.flash_attn_interface import flash_attn_varlen_func


from .utils import create_block_mask_cached, generate_temporal_head_mask_mod
from .placement import hunyuan_sparse_head_placement, hunyuan_hidden_states_placement, ref_hunyuan_sparse_head_placement, ref_hunyuan_hidden_states_placement 

try:
    import flash_attn
    from flash_attn.flash_attn_interface import _flash_attn_forward
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
except ImportError:
    flash_attn = None
    flash_attn_varlen_func = None
    _flash_attn_forward = None


flex_attention = torch.compile(flex_attention, dynamic=False)
torch._dynamo.config.cache_size_limit = 192 * 3
torch._dynamo.config.accumulated_cache_size_limit = 192 * 3


MEMORY_LAYOUT = {
    "flash": (
        lambda x: x.view(x.shape[0] * x.shape[1], *x.shape[2:]),
        lambda x: x,
    ),
    "sparse": (
        lambda x: x.transpose(1, 2).contiguous(),
        lambda x: x.transpose(1, 2).contiguous(),
    ),
    "torch": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
    "vanilla": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
}

class Hunyuan_SparseAttn:
    num_sampled_rows = 32
    attention_masks = None

    context_length = 256
    num_frame = 33
    frame_size = 3600

    first_layers_fp = 0
    first_times_fp = 0

    sample_mse_max_row = 10000
    block_mask = None
    

    def __init__(self):  
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Hunyuan_SparseAttn requires PyTorch 2.0, please upgrade PyTorch.")

    @classmethod
    def sample_mse(self, query, key, value):
        assert len(self.attention_masks) == 2

        cfg, num_heads, seq_len, dim = query.size()
        num_sampled_rows = min(self.num_sampled_rows, seq_len)
        sampled_rows = torch.randint(low=0, high=self.sample_mse_max_row, size=(num_sampled_rows,))
        sampled_q = query[:, :, sampled_rows, :]
        sampled_qk_scores = torch.matmul(sampled_q, key.transpose(-2, -1)) / (dim**0.5)
    
        sampled_attn_weights = F.softmax(sampled_qk_scores, dim=-1)
        sampled_golden_hidden_states = torch.matmul(sampled_attn_weights, value)  # (1, seq_len, dim)

        sampled_mses = torch.zeros(len(self.attention_masks), cfg, num_heads, device=query.device, dtype=query.dtype)

        # Only have Tri-diagonal and Striped
        for mask_idx, attn_mask in enumerate(self.attention_masks):
            sampled_attention_mask = attn_mask[sampled_rows, :]
            sampled_attention_scores = sampled_qk_scores.masked_fill(sampled_attention_mask == 0, float('-inf'))
            sampled_attn_weights = F.softmax(sampled_attention_scores, dim=-1)
            sampled_hidden_states = torch.matmul(sampled_attn_weights, value)
            mse = torch.mean((sampled_hidden_states - sampled_golden_hidden_states) ** 2, dim=(2, 3))
            sampled_mses[mask_idx] = mse

        return sampled_mses

    @classmethod
    def sparse_flex_attention(self, query, key, value, block_mask):
        return flex_attention(query, key, value, block_mask=block_mask)

    @classmethod
    def sparse_head_placement(self, query, key, value, query_out, key_out, value_out, best_mask_idx, context_length, num_frame, frame_size):
        
        query_out, key_out, value_out = ref_hunyuan_sparse_head_placement(query, key, value, best_mask_idx, context_length, num_frame, frame_size)

        return query_out, key_out, value_out

    @classmethod
    def fast_sparse_head_placement(self, query, key, value, query_out, key_out, value_out, best_mask_idx, context_length, num_frame, frame_size):

        hunyuan_sparse_head_placement(query, key, value, query_out, key_out, value_out, best_mask_idx, context_length, num_frame, frame_size)

        return query_out, key_out, value_out

    @classmethod
    def hidden_states_placement(self, \
        hidden_states, output_hidden_states, \
        best_mask_idx, context_length, num_frame, frame_size
    ):
        ref_hunyuan_hidden_states_placement(hidden_states, output_hidden_states, best_mask_idx, context_length, num_frame, frame_size)

    @classmethod
    def fast_hidden_states_placement(self, \
        hidden_states, output_hidden_states, \
        best_mask_idx, context_length, num_frame, frame_size
    ):
        hunyuan_hidden_states_placement(hidden_states, output_hidden_states, best_mask_idx, context_length, num_frame, frame_size)

    @classmethod
    def attention_core_logic(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        timestep,
        layer_idx,
        cu_seqlens_q,
        cu_seqlens_kv,
        max_seqlen_q,
        max_seqlen_kv,
    ):
        cfg, num_heads, seq_len, dim = query.size()
        
        context_length, num_frame, frame_size = self.context_length, self.num_frame, self.frame_size

        assert seq_len == context_length + num_frame * frame_size, \
            f"Query Shape: {seq_len} is not equivalent to {context_length} + {num_frame} * {frame_size}"

        sampled_mses = self.sample_mse(query, key, value)
        best_mask_idx = torch.argmin(sampled_mses, dim=0)


        output_hidden_states = torch.zeros_like(query)

        query_out, key_out, value_out = torch.zeros_like(query), torch.zeros_like(key), torch.zeros_like(value)

        query_out, key_out, value_out = self.fast_sparse_head_placement(query, key, value, query_out, key_out, value_out, best_mask_idx, context_length, num_frame, frame_size)

        hidden_states = self.sparse_flex_attention(query_out, key_out, value_out, block_mask=self.block_mask)

        self.fast_hidden_states_placement(hidden_states, output_hidden_states, best_mask_idx, context_length, num_frame, frame_size)

        return output_hidden_states.reshape(cfg, num_heads, seq_len, dim)


def get_cu_seqlens(text_mask, img_len):
    """Calculate cu_seqlens_q, cu_seqlens_kv using text_mask and img_len

    Args:
        text_mask (torch.Tensor): the mask of text
        img_len (int): the length of image

    Returns:
        torch.Tensor: the calculated cu_seqlens for flash attention
    """
    batch_size = text_mask.shape[0]
    text_len = text_mask.sum(dim=1)
    max_len = text_mask.shape[1] + img_len

    cu_seqlens = torch.zeros([2 * batch_size + 1], dtype=torch.int32, device="cuda")

    for i in range(batch_size):
        s = text_len[i] + img_len
        s1 = i * max_len + s
        s2 = (i + 1) * max_len
        cu_seqlens[2 * i + 1] = s1
        cu_seqlens[2 * i + 2] = s2

    return cu_seqlens


def attention(
    q,
    k,
    v,
    mode="flash",
    drop_rate=0,
    attn_mask=None,
    causal=False,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    max_seqlen_q=None,
    max_seqlen_kv=None,
    batch_size=1,
    timestep=None,
    layer_idx=None
):
    """
    Perform QKV self attention.

    Args:
        q (torch.Tensor): Query tensor with shape [b, s, a, d], where a is the number of heads.
        k (torch.Tensor): Key tensor with shape [b, s1, a, d]
        v (torch.Tensor): Value tensor with shape [b, s1, a, d]
        mode (str): Attention mode. Choose from 'self_flash', 'cross_flash', 'torch', and 'vanilla'.
        drop_rate (float): Dropout rate in attention map. (default: 0)
        attn_mask (torch.Tensor): Attention mask with shape [b, s1] (cross_attn), or [b, a, s, s1] (torch or vanilla).
            (default: None)
        causal (bool): Whether to use causal attention. (default: False)
        cu_seqlens_q (torch.Tensor): dtype torch.int32. The cumulative sequence lengths of the sequences in the batch,
            used to index into q.
        cu_seqlens_kv (torch.Tensor): dtype torch.int32. The cumulative sequence lengths of the sequences in the batch,
            used to index into kv.
        max_seqlen_q (int): The maximum sequence length in the batch of q.
        max_seqlen_kv (int): The maximum sequence length in the batch of k and v.

    Returns:
        torch.Tensor: Output tensor after self attention with shape [b, s, ad]
    """

    # Some Preprocess
    if mode == "sparse":
        assert torch.allclose(cu_seqlens_q, cu_seqlens_kv)
        assert cu_seqlens_kv is not None
                
        # Determine if we use Full Attention to calculate  # TODO  
        full_attention_flag = False
        if layer_idx < 42 * Hunyuan_SparseAttn.first_layers_fp:
            full_attention_flag = True
        if timestep > 1000 * (1 - Hunyuan_SparseAttn.first_times_fp):
            full_attention_flag = True

        if full_attention_flag:    
            mode = "flash"
        else:
            mode = "sparse"

    pre_attn_layout, post_attn_layout = MEMORY_LAYOUT[mode]
    q = pre_attn_layout(q)
    k = pre_attn_layout(k)
    v = pre_attn_layout(v)

    if mode == "torch":
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(q.dtype)
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=drop_rate, is_causal=causal
        )
    elif mode == "sparse":
        x = Hunyuan_SparseAttn.attention_core_logic(
            q, k, v, timestep, layer_idx,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
        )
    elif mode == "flash":
        x = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
        )
        # x with shape [(bxs), a, d]
        x = x.view(
            batch_size, max_seqlen_q, x.shape[-2], x.shape[-1]
        )  # reshape x to [b, s, a, d]
    elif mode == "vanilla":
        scale_factor = 1 / math.sqrt(q.size(-1))

        b, a, s, _ = q.shape
        s1 = k.size(2)
        attn_bias = torch.zeros(b, a, s, s1, dtype=q.dtype, device=q.device)
        if causal:
            # Only applied to self attention
            assert (
                attn_mask is None
            ), "Causal mask and attn_mask cannot be used together"
            temp_mask = torch.ones(b, a, s, s, dtype=torch.bool, device=q.device).tril(
                diagonal=0
            )
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(q.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask

        # TODO: Maybe force q and k to be float32 to avoid numerical overflow
        attn = (q @ k.transpose(-2, -1)) * scale_factor
        attn += attn_bias
        attn = attn.softmax(dim=-1)
        attn = torch.dropout(attn, p=drop_rate, train=True)
        x = attn @ v
    else:
        raise NotImplementedError(f"Unsupported attention mode: {mode}")

    x = post_attn_layout(x)
    b, s, a, d = x.shape
    out = x.reshape(b, s, -1)

    return out


def parallel_attention(
    hybrid_seq_parallel_attn,
    q,
    k,
    v,
    img_q_len,
    img_kv_len,
    cu_seqlens_q,
    cu_seqlens_kv
):
    attn1 = hybrid_seq_parallel_attn(
        None,
        q[:, :img_q_len, :, :],
        k[:, :img_kv_len, :, :],
        v[:, :img_kv_len, :, :],
        dropout_p=0.0,
        causal=False,
        joint_tensor_query=q[:,img_q_len:cu_seqlens_q[1]],
        joint_tensor_key=k[:,img_kv_len:cu_seqlens_kv[1]],
        joint_tensor_value=v[:,img_kv_len:cu_seqlens_kv[1]],
        joint_strategy="rear",
    )
    if flash_attn.__version__ >= '2.7.0':
        attn2, *_ = _flash_attn_forward(
            q[:,cu_seqlens_q[1]:],
            k[:,cu_seqlens_kv[1]:],
            v[:,cu_seqlens_kv[1]:],
            dropout_p=0.0,
            softmax_scale=q.shape[-1] ** (-0.5),
            causal=False,
            window_size_left=-1,
            window_size_right=-1,
            softcap=0.0,
            alibi_slopes=None,
            return_softmax=False,
        )
    else:
        attn2, *_ = _flash_attn_forward(
            q[:,cu_seqlens_q[1]:],
            k[:,cu_seqlens_kv[1]:],
            v[:,cu_seqlens_kv[1]:],
            dropout_p=0.0,
            softmax_scale=q.shape[-1] ** (-0.5),
            causal=False,
            window_size=(-1, -1),
            softcap=0.0,
            alibi_slopes=None,
            return_softmax=False,
        )
    attn = torch.cat([attn1, attn2], dim=1)
    b, s, a, d = attn.shape
    attn = attn.reshape(b, s, -1)

    return attn


def prepare_flexattention(cfg_size, num_head, head_dim, dtype, device, context_length, prompt_length, num_frame, frame_size, \
    diag_width=1, multiplier=2
):
    assert diag_width == multiplier
    seq_len = context_length + num_frame * frame_size
    query, key, value = [torch.zeros((1, cfg_size * num_head, seq_len, head_dim), dtype=dtype, device=device) for _ in range(3)]

    mask_mod = generate_temporal_head_mask_mod(context_length, prompt_length, num_frame, frame_size, mul=multiplier)
    block_mask = create_block_mask_cached(mask_mod, None, None, seq_len, seq_len, device=device, _compile=True)

    hidden_states = flex_attention(query, key, value, block_mask=block_mask)

    return block_mask
