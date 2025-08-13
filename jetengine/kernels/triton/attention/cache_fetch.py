import torch
import triton
import triton.language as tl

# Migrated from layers.cache_fetch (v2)

@triton.jit
def _preprocess_kv_cache_kernel_v2(
    K, V, K_cache, V_cache,
    BLOCK_TABLES, CU_SEQLENS_K, NEW_CU_SEQLENS_K,
    TOKEN_TO_SEQ_IDX_MAP,
    K_full, V_full,
    block_len, block_size,
    num_kv_heads, head_dim,
    stride_k_seq, stride_k_head,
    stride_v_seq, stride_v_head,
    stride_k_cache_block, stride_k_cache_token, stride_k_cache_head,
    stride_v_cache_block, stride_v_cache_token, stride_v_cache_head,
    stride_k_full_seq, stride_k_full_head,
    stride_v_full_seq, stride_v_full_head,
    stride_bt_seq,
    BLOCK_D: tl.constexpr,
):
    pid_token = tl.program_id(0)
    pid_head = tl.program_id(1)
    seq_idx = tl.load(TOKEN_TO_SEQ_IDX_MAP + pid_token)
    new_seq_start = tl.load(NEW_CU_SEQLENS_K + seq_idx)
    token_idx_in_seq = pid_token - new_seq_start
    past_seq_start = tl.load(CU_SEQLENS_K + seq_idx)
    past_seq_end = tl.load(CU_SEQLENS_K + seq_idx + 1)
    past_len = past_seq_end - past_seq_start
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < head_dim
    if token_idx_in_seq < past_len:
        block_idx_in_seq = token_idx_in_seq // block_size
        offset_in_block = token_idx_in_seq % block_size
        bt_ptr = BLOCK_TABLES + seq_idx * stride_bt_seq + block_idx_in_seq
        physical_block_id = tl.load(bt_ptr)
        k_src_ptr = (K_cache + physical_block_id * stride_k_cache_block
                     + offset_in_block * stride_k_cache_token
                     + pid_head * stride_k_cache_head + offs_d)
        v_src_ptr = (V_cache + physical_block_id * stride_v_cache_block
                     + offset_in_block * stride_v_cache_token
                     + pid_head * stride_v_cache_head + offs_d)
    else:
        token_idx_in_curr = token_idx_in_seq - past_len
        src_idx = seq_idx * block_len + token_idx_in_curr
        k_src_ptr = K + src_idx * stride_k_seq + pid_head * stride_k_head + offs_d
        v_src_ptr = V + src_idx * stride_v_seq + pid_head * stride_v_head + offs_d
    k_vec = tl.load(k_src_ptr, mask=mask_d, other=0.0)
    v_vec = tl.load(v_src_ptr, mask=mask_d, other=0.0)
    k_dst_ptr = K_full + pid_token * stride_k_full_seq + pid_head * stride_k_full_head + offs_d
    v_dst_ptr = V_full + pid_token * stride_v_full_seq + pid_head * stride_v_full_head + offs_d
    tl.store(k_dst_ptr, k_vec, mask=mask_d)
    tl.store(v_dst_ptr, v_vec, mask=mask_d)


def preprocess_kv_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    new_cu_seqlens_k: torch.Tensor,
    block_len: int,
    block_size: int,
):
    """Preprocess cached + current KV into contiguous (total_tokens, num_kv_heads, head_dim).
    Args mirror legacy wrapper. Returns (k_full, v_full)."""
    num_seqs, _ = block_tables.shape
    total_tokens = new_cu_seqlens_k[-1].item()
    _, _, num_kv_heads, head_dim = k_cache.shape
    k_full = torch.empty(total_tokens, num_kv_heads, head_dim, dtype=k.dtype, device=k.device)
    v_full = torch.empty_like(k_full, dtype=v.dtype)
    token_indices = torch.arange(total_tokens, device=k.device)
    token_to_seq_idx_map = torch.searchsorted(new_cu_seqlens_k, token_indices, right=True) - 1
    k = k.view(num_seqs * block_len, num_kv_heads, head_dim)
    v = v.view_as(k)
    BLOCK_D = triton.next_power_of_2(head_dim)
    grid = (total_tokens, num_kv_heads)
    _preprocess_kv_cache_kernel_v2[grid](
        K=k, V=v, K_cache=k_cache, V_cache=v_cache,
        BLOCK_TABLES=block_tables,
        CU_SEQLENS_K=cu_seqlens_k,
        NEW_CU_SEQLENS_K=new_cu_seqlens_k,
        TOKEN_TO_SEQ_IDX_MAP=token_to_seq_idx_map,
        K_full=k_full, V_full=v_full,
        block_len=block_len, block_size=block_size,
        num_kv_heads=num_kv_heads, head_dim=head_dim,
        stride_k_seq=k.stride(0), stride_k_head=k.stride(1),
        stride_v_seq=v.stride(0), stride_v_head=v.stride(1),
        stride_k_cache_block=k_cache.stride(0), stride_k_cache_token=k_cache.stride(1), stride_k_cache_head=k_cache.stride(2),
        stride_v_cache_block=v_cache.stride(0), stride_v_cache_token=v_cache.stride(1), stride_v_cache_head=v_cache.stride(2),
        stride_k_full_seq=k_full.stride(0), stride_k_full_head=k_full.stride(1),
        stride_v_full_seq=v_full.stride(0), stride_v_full_head=v_full.stride(1),
        stride_bt_seq=block_tables.stride(0),
        BLOCK_D=BLOCK_D,
    )
    return k_full, v_full

__all__ = ["preprocess_kv_cache"]
