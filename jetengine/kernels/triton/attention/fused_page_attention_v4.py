import torch
import triton
import triton.language as tl
import math
from typing import Optional
import triton.testing

LOG2E = 1.4426950408889634

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'num_warps': 4, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'num_warps': 4, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'num_warps': 4, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'num_warps': 8, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'num_warps': 8, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'num_warps': 8, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'num_warps': 4, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'num_warps': 4, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'num_warps': 4, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'num_warps': 8, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'num_warps': 8, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'num_warps': 8, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'num_warps': 4, 'num_stages': 3}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'num_warps': 4, 'num_stages': 3}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'num_warps': 8, 'num_stages': 3}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'num_warps': 8, 'num_stages': 3}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'num_warps': 4, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'num_warps': 4, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'num_warps': 8, 'num_stages': 3}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'num_warps': 8, 'num_stages': 3}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'num_warps': 2, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'num_warps': 2, 'num_stages': 3}),
    ],
    key=['num_q_heads', 'num_kv_heads', 'head_dim', 'block_len', 'block_size']
)
@triton.jit
def fused_kv_cache_attention_kernel(
    q_ptr, q_stride_seq, q_stride_head, q_stride_dim,
    k_ptr, k_stride_seq, k_stride_head, k_stride_dim,
    v_ptr, v_stride_seq, v_stride_head, v_stride_dim,
    k_cache_ptr, k_cache_stride_block, k_cache_stride_pos, k_cache_stride_head, k_cache_stride_dim,
    v_cache_ptr, v_cache_stride_block, v_cache_stride_pos, v_cache_stride_head, v_cache_stride_dim,
    block_tables_ptr, block_tables_stride_seq, block_tables_stride_block,
    o_ptr, o_stride_seq, o_stride_head, o_stride_dim,
    cu_seqlens_q_ptr, cu_seqlens_k_ptr,
    num_seqs, block_len, block_size, num_q_heads, num_kv_heads, head_dim,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    if seq_idx >= num_seqs:
        return

    seq_start_q = tl.load(cu_seqlens_q_ptr + seq_idx)
    seq_start_k_cache = tl.load(cu_seqlens_k_ptr + seq_idx)
    seq_end_k_cache = tl.load(cu_seqlens_k_ptr + seq_idx + 1)
    seq_len_k_cache = seq_end_k_cache - seq_start_k_cache

    kv_head_idx = head_idx // (num_q_heads // num_kv_heads)

    dim_mask = tl.arange(0, BLOCK_DMODEL) < head_dim

    for start_m in range(0, block_len, BLOCK_M):
        m_i = tl.full([BLOCK_M], value=-float('inf'), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

        q_tile_offsets = start_m + tl.arange(0, BLOCK_M)
        q_mask = q_tile_offsets < block_len
        q_abs_offsets = (seq_start_q + q_tile_offsets[:, None]) * q_stride_seq
        q_ptrs = q_ptr + q_abs_offsets + head_idx * q_stride_head + \
                 tl.arange(0, BLOCK_DMODEL)[None, :] * q_stride_dim
        q_load_mask = q_mask[:, None] & dim_mask[None, :]
        q = tl.load(q_ptrs, mask=q_load_mask, other=0.0).to(tl.float32)
        if seq_len_k_cache > 0:
            num_logical_blocks = tl.cdiv(seq_len_k_cache, block_size)
            kv_pos = 0
            
            for logical_block_num in range(num_logical_blocks):
                physical_block_id = tl.load(block_tables_ptr + seq_idx * block_tables_stride_seq + 
                                          logical_block_num * block_tables_stride_block)
                num_tokens_in_block = tl.minimum(seq_len_k_cache - kv_pos, block_size)

                k_cache_base = k_cache_ptr + physical_block_id * k_cache_stride_block + \
                              kv_head_idx * k_cache_stride_head
                v_cache_base = v_cache_ptr + physical_block_id * v_cache_stride_block + \
                              kv_head_idx * v_cache_stride_head

                for block_start_n in range(0, num_tokens_in_block, BLOCK_N):
                    actual_tile_size = tl.minimum(num_tokens_in_block - block_start_n, BLOCK_N)
                    n_range = tl.arange(0, BLOCK_N)
                    kv_tile_mask = n_range < actual_tile_size
                    
                    k_offsets = (block_start_n + n_range)[:, None] * k_cache_stride_pos
                    k_ptrs = k_cache_base + k_offsets + \
                            tl.arange(0, BLOCK_DMODEL)[None, :] * k_cache_stride_dim
                    k_load_mask = kv_tile_mask[:, None] & dim_mask[None, :]
                    k = tl.load(k_ptrs, mask=k_load_mask, other=0.0).to(tl.float32)
                    
                    qk = tl.dot(q, tl.trans(k)) * sm_scale
                    qk = tl.where(kv_tile_mask[None, :], qk, -float('inf'))

                    m_ij = tl.max(qk, axis=1)
                    m_ij_new = tl.maximum(m_i, m_ij)
                    alpha = tl.exp2(m_i - m_ij_new)
                    p = tl.exp2(qk - m_ij_new[:, None])

                    v_ptrs = v_cache_base + k_offsets + \
                            tl.arange(0, BLOCK_DMODEL)[None, :] * v_cache_stride_dim
                    v = tl.load(v_ptrs, mask=k_load_mask, other=0.0).to(tl.float32)

                    acc = acc * alpha[:, None] + tl.dot(p, v)
                    l_i = l_i * alpha + tl.sum(p, axis=1)
                    m_i = m_ij_new

                kv_pos += num_tokens_in_block

        for kv_block_idx in range(0, block_len, BLOCK_N):
            actual_block_size = tl.minimum(kv_block_idx + BLOCK_N, block_len) - kv_block_idx
            n_range = tl.arange(0, BLOCK_N)
            kv_mask = n_range < actual_block_size
            
            k_abs_offsets = (seq_start_q + kv_block_idx + n_range[:, None]) * k_stride_seq
            k_ptrs = k_ptr + k_abs_offsets + kv_head_idx * k_stride_head + \
                     tl.arange(0, BLOCK_DMODEL)[None, :] * k_stride_dim
            k_load_mask = kv_mask[:, None] & dim_mask[None, :]
            k = tl.load(k_ptrs, mask=k_load_mask, other=0.0).to(tl.float32)
            
            qk = tl.dot(q, tl.trans(k)) * sm_scale
            qk = tl.where(kv_mask[None, :], qk, -float('inf'))
            
            m_ij = tl.max(qk, axis=1)
            m_ij_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp2(m_i - m_ij_new)
            p = tl.exp2(qk - m_ij_new[:, None])
            
            v_abs_offsets = (seq_start_q + kv_block_idx + n_range[:, None]) * v_stride_seq
            v_ptrs = v_ptr + v_abs_offsets + kv_head_idx * v_stride_head + \
                     tl.arange(0, BLOCK_DMODEL)[None, :] * v_stride_dim
            v = tl.load(v_ptrs, mask=k_load_mask, other=0.0).to(tl.float32)
            
            acc = acc * alpha[:, None] + tl.dot(p, v)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_ij_new

        l_i_safe = tl.where(l_i > 0, l_i, 1.0)
        acc = acc / l_i_safe[:, None]
        
        o_abs_offsets = (seq_start_q + q_tile_offsets[:, None]) * o_stride_seq
        o_ptrs = o_ptr + o_abs_offsets + head_idx * o_stride_head + \
                 tl.arange(0, BLOCK_DMODEL)[None, :] * o_stride_dim
        tl.store(o_ptrs, acc.to(o_ptr.dtype.element_ty), mask=q_load_mask)


def fused_kv_cache_attention(
    q: torch.Tensor,  # [num_seqs * block_len, num_q_heads, head_dim]
    k: torch.Tensor,  # [num_seqs * block_len, num_kv_heads, head_dim]
    v: torch.Tensor,  # [num_seqs * block_len, num_kv_heads, head_dim]
    k_cache: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, head_dim]
    block_tables: torch.Tensor,  # [num_seqs, max_blocks_per_seq]
    cu_seqlens_q: torch.Tensor,  # [num_seqs + 1]
    cu_seqlens_k: torch.Tensor,  # [num_seqs + 1]
    block_len: int,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    q_total_tokens, num_q_heads, head_dim = q.shape
    _, num_kv_heads, _ = k.shape
    _, block_size, _, _ = k_cache.shape
    num_seqs = block_tables.shape[0]

    assert cu_seqlens_q.shape[0] == num_seqs + 1
    assert cu_seqlens_k.shape[0] == num_seqs + 1

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim) * LOG2E

    o = torch.empty_like(q)
    
    grid = (num_seqs, num_q_heads)

    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()
    if not v.is_contiguous():
        v = v.contiguous()

    fused_kv_cache_attention_kernel[grid](
        q, q.stride(0), q.stride(1), q.stride(2),
        k, k.stride(0), k.stride(1), k.stride(2),
        v, v.stride(0), v.stride(1), v.stride(2),
        k_cache, k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache, v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        block_tables, block_tables.stride(0), block_tables.stride(1),
        o, o.stride(0), o.stride(1), o.stride(2),
        cu_seqlens_q, cu_seqlens_k,
        num_seqs, block_len, block_size, num_q_heads, num_kv_heads, head_dim,
        float(sm_scale),
        BLOCK_DMODEL=head_dim,
    )

    return o


# Enhanced testing function with benchmarking
def test_optimized_fused_kv_cache_attention():
    torch.manual_seed(42)
    num_seqs = 4
    block_len = 4
    block_size = 256
    num_q_heads = 16
    num_kv_heads = 8
    head_dim = 128
    num_blocks = 128
    max_blocks_per_seq = 32

    device = 'cuda'
    dtype = torch.float16

    total_tokens = num_seqs * block_len
    q = torch.randn(total_tokens, num_q_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(total_tokens, num_kv_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(total_tokens, num_kv_heads, head_dim, device=device, dtype=dtype)

    k_cache = torch.randn(num_blocks, num_kv_heads, head_dim, block_size, device=device, dtype=dtype).permute(0, 3, 1, 2).contiguous()
    v_cache = torch.randn(num_blocks, num_kv_heads, head_dim, block_size, device=device, dtype=dtype).permute(0, 3, 1, 2).contiguous()

    block_tables = torch.randint(0, num_blocks, (num_seqs, max_blocks_per_seq), device=device, dtype=torch.int32)

    cached_lens = torch.tensor([48, 31, 64, 17], device=device)
    cu_seqlens_k = torch.cumsum(torch.cat([torch.tensor([0], device=device), cached_lens]), dim=0).to(torch.int32)
    cu_seqlens_q = torch.arange(0, (num_seqs + 1) * block_len, block_len, device=device, dtype=torch.int32)

    def reference_implementation():
        outputs = []
        scale = 1.0 / math.sqrt(head_dim)
        for seq_idx in range(num_seqs):
            q_start, q_end = cu_seqlens_q[seq_idx].item(), cu_seqlens_q[seq_idx+1].item()
            cache_len = cached_lens[seq_idx].item()
            q_seq = q[q_start:q_end]
            k_seq = k[q_start:q_end]
            v_seq = v[q_start:q_end]
            k_cached_full = []
            v_cached_full = []
            if cache_len > 0:
                num_cache_blocks = (cache_len + block_size - 1) // block_size
                rem_len = cache_len
                for i in range(num_cache_blocks):
                    physical_block_id = block_tables[seq_idx, i].item()
                    len_to_get = min(rem_len, block_size)
                    k_cached_full.append(k_cache[physical_block_id, :len_to_get])
                    v_cached_full.append(v_cache[physical_block_id, :len_to_get])
                    rem_len -= len_to_get
            if k_cached_full:
                k_full = torch.cat(k_cached_full + [k_seq], dim=0)
                v_full = torch.cat(v_cached_full + [v_seq], dim=0)
            else:
                k_full = k_seq
                v_full = v_seq
            q_in = q_seq.unsqueeze(0).transpose(1, 2)
            k_in = k_full.unsqueeze(0).transpose(1, 2)
            v_in = v_full.unsqueeze(0).transpose(1, 2)
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q_in, k_in, v_in, scale=scale, enable_gqa=True, is_causal=False
            )
            outputs.append(attn_output.transpose(1, 2).squeeze(0))
        return torch.cat(outputs, dim=0)

    # Test correctness
    ref_output = reference_implementation()
    optimized_output = fused_kv_cache_attention(
        q, k, v, k_cache, v_cache, block_tables, cu_seqlens_q, cu_seqlens_k, block_len
    )

    rtol = 1e-2
    atol = 1e-2
    assert torch.allclose(ref_output, optimized_output, rtol=rtol, atol=atol), "Correctness test failed"

    print('Optimized kernel test passed')

    # Benchmark if triton.testing is available
    try:
        print("\nBenchmarking...")
        ms_optimized = triton.testing.do_bench(
            lambda: optimized_fused_kv_cache_attention(
                q, k, v, k_cache, v_cache, block_tables, cu_seqlens_q, cu_seqlens_k, block_len
            )
        )
        print(f"Optimized kernel: {ms_optimized:.3f} ms")
        
        ms_reference = triton.testing.do_bench(lambda: reference_implementation())
        print(f"Reference implementation: {ms_reference:.3f} ms")
        print(f"Speedup: {ms_reference / ms_optimized:.2f}x")
        
    except Exception as e:
        print(f"Benchmarking not available: {e}")


if __name__ == '__main__':
    test_optimized_fused_kv_cache_attention()