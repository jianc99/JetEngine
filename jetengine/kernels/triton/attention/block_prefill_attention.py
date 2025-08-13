import math
import torch
import triton
import triton.language as tl

# Migrated from layers.block_prefill_attention (v1)

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=3, num_warps=8),
    ],
    key=["BLOCK_DMODEL", "STAIRCASE_SIZE"],
)
@triton.jit
def _staircase_attn_fwd_kernel_varlen(
    Q, K, V, Out,
    cu_seqlens_q, cu_seqlens_k,
    stride_qt, stride_qh, stride_qk,
    stride_kt, stride_kh, stride_kk,
    stride_vt, stride_vh, stride_vk,
    stride_ot, stride_oh, stride_ok,
    n_heads, n_kv_heads,
    q_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    STAIRCASE_SIZE: tl.constexpr,
    NUM_HEADS_PER_KV_GROUP: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    kv_head_idx = head_idx // NUM_HEADS_PER_KV_GROUP
    q_seq_start = tl.load(cu_seqlens_q + seq_idx).to(tl.int32)
    q_seq_end = tl.load(cu_seqlens_q + seq_idx + 1).to(tl.int32)
    k_seq_start = tl.load(cu_seqlens_k + seq_idx).to(tl.int32)
    k_seq_end = tl.load(cu_seqlens_k + seq_idx + 1).to(tl.int32)
    q_seq_len = q_seq_end - q_seq_start
    k_seq_len = k_seq_end - k_seq_start
    if q_seq_len == 0:
        return
    for start_m in range(0, q_seq_len, BLOCK_M):
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        q_block_abs_start = q_seq_start + start_m
        Q_block_ptr = tl.make_block_ptr(
            base=Q + head_idx * stride_qh,
            shape=(q_seq_end, BLOCK_DMODEL),
            strides=(stride_qt, stride_qk),
            offsets=(q_block_abs_start, 0),
            block_shape=(BLOCK_M, BLOCK_DMODEL),
            order=(1, 0),
        )
        q = tl.load(Q_block_ptr, boundary_check=(0,))
        q = (q * q_scale).to(q.type)
        q_block_rel_end = tl.minimum(start_m + BLOCK_M - 1, q_seq_len - 1)
        max_staircase_block = q_block_rel_end // STAIRCASE_SIZE
        end_n = (max_staircase_block + 1) * STAIRCASE_SIZE
        end_n = tl.minimum(end_n, k_seq_len)
        for start_n in range(0, end_n, BLOCK_N):
            k_block_abs_start = k_seq_start + start_n
            K_iter_ptr = tl.make_block_ptr(
                base=K + kv_head_idx * stride_kh,
                shape=(k_seq_end, BLOCK_DMODEL),
                strides=(stride_kt, stride_kk),
                offsets=(k_block_abs_start, 0),
                block_shape=(BLOCK_N, BLOCK_DMODEL),
                order=(1, 0),
            )
            V_iter_ptr = tl.make_block_ptr(
                base=V + kv_head_idx * stride_vh,
                shape=(k_seq_end, BLOCK_DMODEL),
                strides=(stride_vt, stride_vk),
                offsets=(k_block_abs_start, 0),
                block_shape=(BLOCK_N, BLOCK_DMODEL),
                order=(1, 0),
            )
            k = tl.load(K_iter_ptr, boundary_check=(0,))
            v = tl.load(V_iter_ptr, boundary_check=(0,))
            qk = tl.dot(q, tl.trans(k))
            offs_m = start_m + tl.arange(0, BLOCK_M)
            offs_n = start_n + tl.arange(0, BLOCK_N)
            m_valid = offs_m < q_seq_len
            n_valid = offs_n < k_seq_len
            m_staircase = offs_m // STAIRCASE_SIZE
            n_staircase = offs_n // STAIRCASE_SIZE
            mask = ((m_staircase[:, None] >= n_staircase[None, :])) & m_valid[:, None] & n_valid[None, :]
            qk = tl.where(mask, qk, -1.0e6)
            m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
            alpha = tl.exp2(m_i - m_i_new)
            p = tl.exp2(qk - m_i_new[:, None])
            l_ij = tl.sum(p, axis=1)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]
            m_i = m_i_new
            acc = tl.dot(p.to(v.type), v, acc)
        l_i_safe = tl.where(l_i == 0, 1.0, l_i)
        acc = acc / l_i_safe[:, None]
        O_block_ptr = tl.make_block_ptr(
            base=Out + head_idx * stride_oh,
            shape=(q_seq_end, BLOCK_DMODEL),
            strides=(stride_ot, stride_ok),
            offsets=(q_block_abs_start, 0),
            block_shape=(BLOCK_M, BLOCK_DMODEL),
            order=(1, 0),
        )
        tl.store(O_block_ptr, acc.to(q.type), boundary_check=(0,))

class SparseAttentionVarlenFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor, staircase_size: int) -> torch.Tensor:
        assert q.dim() == k.dim() == v.dim() == 3
        assert q.shape[0] == k.shape[0] == v.shape[0]
        assert k.shape[1] == v.shape[1] and k.shape[2] == v.shape[2]
        assert q.shape[2] == k.shape[2]
        assert q.shape[1] % k.shape[1] == 0
        assert staircase_size in [1,2,4,8,16]
        total_tokens, n_heads, head_dim = q.shape
        _, n_kv_heads, _ = k.shape
        batch_size = cu_seqlens_q.numel() - 1
        num_heads_per_kv_group = n_heads // n_kv_heads
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        BLOCK_DMODEL = head_dim
        o = torch.empty_like(q)
        grid = (batch_size, n_heads)
        _staircase_attn_fwd_kernel_varlen[grid](
            q, k, v, o,
            cu_seqlens_q, cu_seqlens_k,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            n_heads, n_kv_heads,
            q_scale=1.0 / math.sqrt(head_dim) * 1.44269504,
            BLOCK_DMODEL=BLOCK_DMODEL,
            STAIRCASE_SIZE=staircase_size,
            NUM_HEADS_PER_KV_GROUP=num_heads_per_kv_group,
        )
        return o

def sparse_attn_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, staircase_size: int = 4):
    return SparseAttentionVarlenFunction.apply(q, k, v, cu_seqlens_q, cu_seqlens_k, staircase_size)
