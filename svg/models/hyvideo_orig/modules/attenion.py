import json
import math

import torch
import torch.nn.functional as F
from flash_attn.flash_attn_interface import flash_attn_varlen_func
from torch.nn.attention.flex_attention import flex_attention

from ....kmeans_utils import (
    apply_inverse_permutation,
    density_calculation,
    dynamic_block_sparse_fwd_torch,
    dynamic_block_sparse_fwd_triton,
    dynamic_block_sparse_fwd_flashinfer,
    identify_dynamic_map,
    kmeans_rapidai,
    batch_kmeans_Euclid,
    permute_tensor_by_labels,
)
from ....timer import time_logging_decorator
from .placement import (
    hunyuan_hidden_states_placement,
    hunyuan_sparse_head_placement,
    ref_hunyuan_hidden_states_placement,
    ref_hunyuan_sparse_head_placement,
)
from .utils import create_block_mask_cached, generate_temporal_head_mask_mod

try:
    import flash_attn
    from flash_attn.flash_attn_interface import (
        _flash_attn_forward,
        flash_attn_varlen_func,
    )
except ImportError:
    flash_attn = None
    flash_attn_varlen_func = None
    _flash_attn_forward = None

import flashinfer

flex_attention = torch.compile(flex_attention, dynamic=False)
torch._dynamo.config.cache_size_limit = 192 * 3
torch._dynamo.config.accumulated_cache_size_limit = 192 * 3


MEMORY_LAYOUT = {
    "flash": (
        lambda x: x.view(x.shape[0] * x.shape[1], *x.shape[2:]),
        lambda x: x,
    ),
    "flashinfer": (
        lambda x: x.transpose(1, 2).contiguous(),
        lambda x: x.transpose(1, 2).contiguous(),
    ),
    "SVG": (
        lambda x: x.transpose(1, 2).contiguous(),
        lambda x: x.transpose(1, 2).contiguous(),
    ),
    "SAP": (
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

    prompt_length = 0
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
            sampled_attention_scores = sampled_qk_scores.masked_fill(sampled_attention_mask == 0, float("-inf"))
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
    def hidden_states_placement(self, hidden_states, output_hidden_states, best_mask_idx, context_length, num_frame, frame_size):
        ref_hunyuan_hidden_states_placement(hidden_states, output_hidden_states, best_mask_idx, context_length, num_frame, frame_size)

    @classmethod
    def fast_hidden_states_placement(self, hidden_states, output_hidden_states, best_mask_idx, context_length, num_frame, frame_size):
        hunyuan_hidden_states_placement(hidden_states, output_hidden_states, best_mask_idx, context_length, num_frame, frame_size)

    @classmethod
    def attention_core_logic(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        timestep: torch.Tensor,
        layer_idx: int,
    ):
        cfg, num_heads, seq_len, dim = query.size()

        context_length, num_frame, frame_size = self.context_length, self.num_frame, self.frame_size

        assert seq_len == context_length + num_frame * frame_size, f"Query Shape: {seq_len} is not equivalent to {context_length} + {num_frame} * {frame_size}"

        sampled_mses = self.sample_mse(query, key, value)
        best_mask_idx = torch.argmin(sampled_mses, dim=0)

        output_hidden_states = torch.zeros_like(query)

        query_out, key_out, value_out = torch.zeros_like(query), torch.zeros_like(key), torch.zeros_like(value)

        query_out, key_out, value_out = self.fast_sparse_head_placement(query, key, value, query_out, key_out, value_out, best_mask_idx, context_length, num_frame, frame_size)

        hidden_states = self.sparse_flex_attention(query_out, key_out, value_out, block_mask=self.block_mask)

        self.fast_hidden_states_placement(hidden_states, output_hidden_states, best_mask_idx, context_length, num_frame, frame_size)

        return output_hidden_states.reshape(cfg, num_heads, seq_len, dim)


# ---- Semantic Aware Permutation Processor ----
class Hunyuan_SAPAttn(Hunyuan_SparseAttn):
    """
    Semantic Aware Permutation Attention
    """

    # centroids
    num_q_centroids = 0
    num_k_centroids = 0
    top_p_kmeans = 0
    min_kc_ratio = 0

    centroids_init = {}
    q_centroids = {}
    k_centroids = {}

    kmeans_iter_init = 0
    kmeans_iter_step = 0
    zero_step_kmeans_init = False

    logging_file = None
    
    
    @classmethod
    @time_logging_decorator("Level 3 - kmeans init")
    def kmeans_init(self, query, key, layer_idx):
        cfg, num_heads, seq_len, dim = query.size()
        qlabels, qcentroids, qcluster_sizes, qiter = batch_kmeans_Euclid(query.view(cfg * num_heads, seq_len, dim), n_clusters=self.num_q_centroids, max_iters=self.kmeans_iter_init)
        klabels, kcentroids, kcluster_sizes, kiter = batch_kmeans_Euclid(key.view(cfg * num_heads, seq_len, dim), n_clusters=self.num_k_centroids, max_iters=self.kmeans_iter_init)

        self.q_centroids[layer_idx] = qcentroids
        self.k_centroids[layer_idx] = kcentroids

        return qlabels, qcentroids, qcluster_sizes, qiter, klabels, kcentroids, kcluster_sizes, kiter


    @classmethod
    @time_logging_decorator("Level 3 - kmeans step")
    def kmeans_step(self, query, key, layer_idx):
        cfg, num_heads, seq_len, dim = query.size()
        qlabels, qcentroids, qcluster_sizes, qiter = batch_kmeans_Euclid(query.view(cfg * num_heads, seq_len, dim), n_clusters=self.num_q_centroids, max_iters=self.kmeans_iter_step, init_centroids=self.q_centroids[layer_idx])
        klabels, kcentroids, kcluster_sizes, kiter = batch_kmeans_Euclid(key.view(cfg * num_heads, seq_len, dim), n_clusters=self.num_k_centroids, max_iters=self.kmeans_iter_step, init_centroids=self.k_centroids[layer_idx])

        self.q_centroids[layer_idx] = qcentroids
        self.k_centroids[layer_idx] = kcentroids

        return qlabels, qcentroids, qcluster_sizes, qiter, klabels, kcentroids, kcluster_sizes, kiter

    @classmethod
    @time_logging_decorator("Level 3 - kmeans clustering")
    def kmeans_clustering(self, query, key, layer_idx):
        if layer_idx not in self.centroids_init.keys():
            qlabels, qcentroids, qcluster_sizes, qiter, klabels, kcentroids, kcluster_sizes, kiter = self.kmeans_init(query, key, layer_idx)
            self.centroids_init[layer_idx] = True
            print(f"Centroids initialized at layer {layer_idx}. Init step: {self.kmeans_iter_init}")
        else:
            assert self.centroids_init[layer_idx], f"Centroids not initialized at layer {layer_idx}"
            qlabels, qcentroids, qcluster_sizes, qiter, klabels, kcentroids, kcluster_sizes, kiter = self.kmeans_step(query, key, layer_idx)

        return qlabels, qcentroids, qcluster_sizes, qiter, klabels, kcentroids, kcluster_sizes, kiter

    @classmethod
    @time_logging_decorator("Level 3 - semantic aware permutation")
    def semantic_aware_permutation(self, query, key, value, timestep, layer_idx):
        cfg, num_heads, seq_len, dim = query.size()
        
        # 1. Kmeans clustering
        qlabels, qcentroids, qcluster_sizes, qiter, klabels, kcentroids, kcluster_sizes, kiter = self.kmeans_clustering(query, key, layer_idx)

        # 2. Identify dynamic map
        q_cluster_sizes = qcluster_sizes.view(cfg, num_heads, self.num_q_centroids)
        k_cluster_sizes = kcluster_sizes.view(cfg, num_heads, self.num_k_centroids)

        dynamic_map = identify_dynamic_map(qcentroids.view(cfg, num_heads, self.num_q_centroids, dim), kcentroids.view(cfg, num_heads, self.num_k_centroids, dim), q_cluster_sizes, k_cluster_sizes, self.top_p_kmeans, self.min_kc_ratio)

        # 3. Permute the query, key, value
        q_permuted, q_sorted_indices = permute_tensor_by_labels(query, qlabels, dim=2)
        k_permuted, _ = permute_tensor_by_labels(key, klabels, dim=2)
        v_permuted, _ = permute_tensor_by_labels(value, klabels, dim=2)

        return q_permuted, k_permuted, v_permuted, dynamic_map, q_cluster_sizes, k_cluster_sizes, q_sorted_indices

    @classmethod
    @time_logging_decorator("Level 3 - dynamic map post processing")
    def dynamic_map_post_processing(self, q_perm, k_perm, v_perm, query, key, value, dyn_map, qc_sz_s, kc_sz_s, q_sorted_indices, video_length, context_length, prompt_length, unprompt_length):
        # Update the query, key, value
        q_perm = torch.cat([q_perm, query[:, :, -context_length:, :]], dim=2)
        k_perm = torch.cat([k_perm, key[:, :, -context_length:, :]], dim=2)
        v_perm = torch.cat([v_perm, value[:, :, -context_length:, :]], dim=2)

        # Update the dynamic map
        dyn_map = F.pad(dyn_map, (0, 2, 0, 2), value=0)
        dyn_map[:, :, -2, :-1] = True
        dyn_map[:, :, :-1, -2] = True
        dyn_map[:, :, -1, -1] = True
        
        # Update the cluster sizes
        qc_sz_s = F.pad(qc_sz_s, (0, 2), value=0)
        qc_sz_s[:, :, -2] = prompt_length
        qc_sz_s[:, :, -1] = unprompt_length
        kc_sz_s = F.pad(kc_sz_s, (0, 2), value=0)
        kc_sz_s[:, :, -2] = prompt_length
        kc_sz_s[:, :, -1] = unprompt_length
        
        # Update the sorted indices
        q_sorted_indices = F.pad(q_sorted_indices, (0, context_length), value=0)
        q_sorted_indices[:, video_length:] = torch.arange(video_length, video_length + context_length, device=q_sorted_indices.device)
        
        q_sorted_indices = q_sorted_indices.unsqueeze(0)

        return q_perm, k_perm, v_perm, dyn_map, qc_sz_s, kc_sz_s, q_sorted_indices


    @classmethod
    @time_logging_decorator("Level 2 - attention core logic")
    def attention_core_logic(self, query, key, value, timestep, layer_idx):
        cfg, num_heads, seq_len, dim = query.size()
        assert cfg == 1, f"Batch size must be 1 for kmeans block sparse attention"

        prompt_length, context_length, num_frame, frame_size = self.prompt_length, self.context_length, self.num_frame, self.frame_size

        assert seq_len == context_length + num_frame * frame_size, f"Query Shape: {seq_len} is not equivalent to {context_length} + {num_frame} * {frame_size}"

        # Due to Hunyuan's design, we need to split the query, key, value into 1. video part, 2. prompt part, 3. Unused prompt part
        video_length = num_frame * frame_size
        unprompt_length = context_length - prompt_length

        # 1. Video part
        query_video = query[:, :, :video_length, :].contiguous()
        key_video = key[:, :, :video_length, :].contiguous()
        value_video = value[:, :, :video_length, :].contiguous()
        
        attn_output = torch.zeros_like(query)

        # Core logic
        q_perm, k_perm, v_perm, dyn_map, qc_sz_s, kc_sz_s, q_sorted_indices = self.semantic_aware_permutation(query_video, key_video, value_video, timestep, layer_idx)

        # Post-processing to make sure Part 1 and Part 2 can attend each other, while Part 3 is only attended by itself
        q_perm, k_perm, v_perm, dyn_map, qc_sz_s, kc_sz_s, q_sorted_indices = self.dynamic_map_post_processing(q_perm, k_perm, v_perm, query, key, value, dyn_map, qc_sz_s, kc_sz_s, q_sorted_indices, video_length, context_length, prompt_length, unprompt_length)

        output_permuted = dynamic_block_sparse_fwd_flashinfer(q_perm, k_perm, v_perm, dyn_map, qc_sz_s, kc_sz_s, is_cpu=False)

        attn_output = apply_inverse_permutation(output_permuted, q_sorted_indices, dim=2)

        # Save time, layer, density information to logging file
        if self.logging_file is not None:
            # Create log entry
            densities = density_calculation(dyn_map, qc_sz_s, kc_sz_s)

            avg_density = densities.mean().item()
            log_entry = {"timestep": timestep[0].item(), "layer": layer_idx, "avg_density": avg_density, "density": densities.tolist()}

            # Append to log file
            with open(self.logging_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

        return attn_output.reshape(cfg, num_heads, seq_len, dim)


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


def flashinfer_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv):
    """
    Launcher for the Flashinfer varlen kernel for HunyuanVideo.

    Args:
        q (torch.Tensor): Query tensor, shape [B, H, S, D].
        k (torch.Tensor): Key tensor, shape [B, H, S, D].
        v (torch.Tensor): Value tensor, shape [B, H, S, D].
        block_mask_map (torch.Tensor): Boolean mask, shape [B, H, qc_num, kc_num]. Currently must on CPU.
        block_row_sz (torch.Tensor): Query block sizes, shape [B, H, qc_num]. Currently must on CPU.
        block_col_sz (torch.Tensor): Key block sizes, shape [B, H, kc_num]. Currently must on CPU.
    """

    # Create block mask map
    B, H, S, D = q.shape
    block_row_sz = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    block_col_sz = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
    block_mask_map = torch.tensor([[True, False,], [False, True]], device=q.device, dtype=torch.bool)
    
    # Expand dims
    block_row_sz = block_row_sz.repeat(B, H, 1)
    block_col_sz = block_col_sz.repeat(B, H, 1)
    block_mask_map = block_mask_map.repeat(B, H, 1, 1)
    
    # Input shape check
    qc_num = block_row_sz.shape[-1]
    kc_num = block_col_sz.shape[-1]
    assert block_mask_map.shape == (B, H, qc_num, kc_num)

    # Check if block_col_sz and block_row_sz are the same for each head
    assert torch.all(block_col_sz.sum(dim=2) == block_col_sz.sum(dim=2)[0, 0])
    assert torch.all(block_row_sz.sum(dim=2) == block_row_sz.sum(dim=2)[0, 0])

    # Prepare flashinfer wrapper
    float_workspace_buffer = torch.empty(128 * 1024 * 1024, device=q.device)
    wrapper = flashinfer.sparse.VariableBlockSparseAttentionWrapper(
        float_workspace_buffer, backend="auto"
    )
    
    # Reshape inputs to (B * H, ...)
    q = q.reshape(B * H, S, D)
    k = k.reshape(B * H, S, D)
    v = v.reshape(B * H, S, D)
    block_mask_map = block_mask_map.reshape(B * H, qc_num, kc_num)
    block_row_sz = block_row_sz.reshape(B * H, qc_num)
    block_col_sz = block_col_sz.reshape(B * H, kc_num)

    wrapper.plan(
        block_mask_map=block_mask_map,
        block_row_sz=block_row_sz,
        block_col_sz=block_col_sz,
        num_qo_heads=B * H,
        num_kv_heads=B * H,
        head_dim=D,
        q_data_type=q.dtype,
        kv_data_type=k.dtype,
    )

    o = wrapper.run(q, k, v)  # [num_qo_heads, qo_len, head_dim]
    o = o.reshape(B, H, S, D)
    return o


def attention(
    q, k, v, mode="flash", drop_rate=0, attn_mask=None, causal=False, cu_seqlens_q=None, cu_seqlens_kv=None, max_seqlen_q=None, max_seqlen_kv=None, batch_size=1, timestep=None, layer_idx=None
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
    if mode == "SVG":
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
            mode = "SVG"
    elif mode == "SAP":
        assert torch.allclose(cu_seqlens_q, cu_seqlens_kv)
        assert cu_seqlens_kv is not None

        # Determine if we use Full Attention to calculate  # TODO
        full_attention_flag = False
        if layer_idx < 42 * Hunyuan_SAPAttn.first_layers_fp:
            full_attention_flag = True
        if timestep > 1000 * (1 - Hunyuan_SAPAttn.first_times_fp):
            full_attention_flag = True

        if full_attention_flag:
            mode = "flashinfer"
        else:
            mode = "SAP"

    pre_attn_layout, post_attn_layout = MEMORY_LAYOUT[mode]
    q = pre_attn_layout(q)
    k = pre_attn_layout(k)
    v = pre_attn_layout(v)

    if mode == "torch":
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(q.dtype)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=drop_rate, is_causal=causal)
    elif mode == "SVG":
        x = Hunyuan_SparseAttn.attention_core_logic(
            q,
            k,
            v,
            timestep,
            layer_idx,
        )
    elif mode == "SAP":
        x = Hunyuan_SAPAttn.attention_core_logic(
            q,
            k,
            v,
            timestep,
            layer_idx,
        )
    elif mode == "flashinfer":
        if Hunyuan_SAPAttn.zero_step_kmeans_init:
            video_length = Hunyuan_SAPAttn.num_frame * Hunyuan_SAPAttn.frame_size
            query_video = q[:, :, :video_length, :].contiguous()
            key_video = k[:, :, :video_length, :].contiguous()
            Hunyuan_SAPAttn.kmeans_clustering(query_video, key_video, layer_idx)
            
        x = flashinfer_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv)
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
        x = x.view(batch_size, max_seqlen_q, x.shape[-2], x.shape[-1])  # reshape x to [b, s, a, d]
    elif mode == "vanilla":
        scale_factor = 1 / math.sqrt(q.size(-1))

        b, a, s, _ = q.shape
        s1 = k.size(2)
        attn_bias = torch.zeros(b, a, s, s1, dtype=q.dtype, device=q.device)
        if causal:
            # Only applied to self attention
            assert attn_mask is None, "Causal mask and attn_mask cannot be used together"
            temp_mask = torch.ones(b, a, s, s, dtype=torch.bool, device=q.device).tril(diagonal=0)
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


def parallel_attention(hybrid_seq_parallel_attn, q, k, v, img_q_len, img_kv_len, cu_seqlens_q, cu_seqlens_kv):
    attn1 = hybrid_seq_parallel_attn(
        None,
        q[:, :img_q_len, :, :],
        k[:, :img_kv_len, :, :],
        v[:, :img_kv_len, :, :],
        dropout_p=0.0,
        causal=False,
        joint_tensor_query=q[:, img_q_len : cu_seqlens_q[1]],
        joint_tensor_key=k[:, img_kv_len : cu_seqlens_kv[1]],
        joint_tensor_value=v[:, img_kv_len : cu_seqlens_kv[1]],
        joint_strategy="rear",
    )
    if flash_attn.__version__ >= "2.7.0":
        attn2, *_ = _flash_attn_forward(
            q[:, cu_seqlens_q[1] :],
            k[:, cu_seqlens_kv[1] :],
            v[:, cu_seqlens_kv[1] :],
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
            q[:, cu_seqlens_q[1] :],
            k[:, cu_seqlens_kv[1] :],
            v[:, cu_seqlens_kv[1] :],
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


def prepare_flexattention(cfg_size, num_head, head_dim, dtype, device, context_length, prompt_length, num_frame, frame_size, diag_width=1, multiplier=2):
    assert diag_width == multiplier
    seq_len = context_length + num_frame * frame_size
    query, key, value = [torch.zeros((1, cfg_size * num_head, seq_len, head_dim), dtype=dtype, device=device) for _ in range(3)]

    mask_mod = generate_temporal_head_mask_mod(context_length, prompt_length, num_frame, frame_size, mul=multiplier)
    block_mask = create_block_mask_cached(mask_mod, None, None, seq_len, seq_len, device=device, _compile=True)

    hidden_states = flex_attention(query, key, value, block_mask=block_mask)

    return block_mask
