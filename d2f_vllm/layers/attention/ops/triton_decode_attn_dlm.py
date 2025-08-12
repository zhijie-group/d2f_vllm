# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: D2F

# Organization: SJTU DENG Lab
# Author: Drew Jin (JIN. Yijie, @drewjin)
# Date: 2025-08-07
# Email: drewjin0827@gmail.com
# All rights reserved.

import torch
import triton

import triton.language as tl


@triton.jit
def _dlm_fd_inner_kv_cache(Q_i, K_j, V_j, acc, m_i, l_i, softmax_scale,
                           BLOCK_M: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    S_ij = tl.zeros([BLOCK_M, BLOCK_SIZE], dtype=tl.float32)
    S_ij += tl.dot(Q_i, K_j) * softmax_scale
        
    m_ij = tl.maximum(m_i, tl.max(S_ij, axis=1))
    P_ij = tl.exp(S_ij - m_ij[:, None])
    
    alpha = tl.exp(m_i - m_ij)
    l_ij = alpha * l_i + tl.sum(P_ij, axis=1)
    
    P_ij = P_ij.to(V_j.type.element_ty)
    acc = alpha[:, None] * acc + tl.dot(P_ij, V_j)
    
    l_i = l_ij
    m_i = m_ij
    
    return acc, m_i, l_i


@triton.jit
def _dlm_fd_inner_self(Q_i, K_j, V_j, Mask, acc, m_i, l_i, softmax_scale,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    S_ij = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    S_ij += tl.dot(Q_i, K_j) * softmax_scale
    S_ij += tl.where(Mask, 0.0, float('-inf'))
        
    m_ij = tl.maximum(m_i, tl.max(S_ij, axis=1))
    P_ij = tl.exp(S_ij - m_ij[:, None])
    
    alpha = tl.exp(m_i - m_ij)
    l_ij = alpha * l_i + tl.sum(P_ij, axis=1)
    
    P_ij = P_ij.to(V_j.type.element_ty)
    acc = alpha[:, None] * acc + tl.dot(P_ij, V_j)
    
    l_i = l_ij
    m_i = m_ij
    
    return acc, m_i, l_i


@triton.jit
def dlm_flash_decoding_kernel(q_ptr, k_ptr, v_ptr, o_ptr, mask_ptr, softmax_scale,
                              k_cache_ptr, v_cache_ptr, block_tables_ptr,
                              cu_seqlens_q_ptr, total_lens_ptr, ctx_lens_ptr,
                              q_stride_m, q_stride_nh, q_stride_d,
                              k_stride_n, k_stride_nh, k_stride_d,
                              v_stride_n, v_stride_nh, v_stride_d,
                              o_stride_m, o_stride_nh, o_stride_d,
                              mask_stride_m, mask_stride_n,
                              k_cache_stride_nblks, k_cache_stride_h, k_cache_stride_dx, k_cache_stride_blk_sz, k_cache_stride_x,
                              v_cache_stride_nblks, v_cache_stride_h, v_cache_stride_d, v_cache_stride_blk_sz,
                              block_tables_stride_nseqs, block_tables_stride_nblks,
                              cu_seqlens_q_ptr_stride, total_lens_ptr_stride, ctx_lens_ptr_stride,
                              NUM_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr, KV_HEAD_GROUP_SIZE: tl.constexpr,
                              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, x: tl.constexpr, BLOCK_SIZE: tl.constexpr,
                              NUM_UNROLL_CACHE: tl.constexpr = 4, NUM_UNROLL_Q: tl.constexpr = 1):
    # Load program IDs
    seq_id = tl.program_id(0)
    head_id = tl.program_id(1)
    blk_start_m = tl.program_id(2)
    
    # Load basic parameters
    kv_head_id = head_id // KV_HEAD_GROUP_SIZE
    total_seq_len = tl.load(total_lens_ptr + seq_id * total_lens_ptr_stride)
    ctx_len = tl.load(ctx_lens_ptr + seq_id * ctx_lens_ptr_stride)
    q_start_id = tl.load(cu_seqlens_q_ptr + seq_id * cu_seqlens_q_ptr_stride)
    q_seq_len = total_seq_len - ctx_len
    
    # Init offset tensors
    offs_blk_sz = tl.arange(0, BLOCK_SIZE)
    offs_m = blk_start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_hdim = tl.arange(0, HEAD_DIM)
    
    # Load Q
    mask_q = offs_m[:, None] < q_seq_len
    offs_q = (offs_m[:, None] + q_start_id) * q_stride_m + head_id * q_stride_nh + offs_hdim[None, :] * q_stride_d
    Q = tl.load(q_ptr + offs_q, mask=mask_q, other=0.0)
    
    # Init Cumulators
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], value=-1e9, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    # Compute KV Cache First
    for blk_start_n in tl.range(0, ctx_len, BLOCK_N, loop_unroll_factor=NUM_UNROLL_CACHE):
        blk_start_n = tl.multiple_of(blk_start_n, BLOCK_SIZE)
        off_page_blk = seq_id * block_tables_stride_nseqs + (blk_start_n // BLOCK_SIZE) * block_tables_stride_nblks
        page_blk_id = tl.load(block_tables_ptr + off_page_blk)
        
        mask_k_cache = (offs_blk_sz[None, :] + blk_start_n) < ctx_len
        offs_k_cache = (page_blk_id[None, :] * k_cache_stride_nblks + 
                        kv_head_id * k_cache_stride_h +
                        (offs_hdim[:, None] // x) * k_cache_stride_dx +
                        (offs_blk_sz[None, :] + blk_start_n) * k_cache_stride_blk_sz +
                        (offs_hdim[:, None] % x) * k_cache_stride_x)
        K_Cache = tl.load(k_cache_ptr + offs_k_cache, mask=mask_k_cache, other=0.0)
        
        mask_v_cache = (offs_blk_sz[:, None] + blk_start_n) < ctx_len
        offs_v_cache = (page_blk_id[:, None] * v_cache_stride_nblks +
                        kv_head_id * v_cache_stride_h +
                        offs_hdim[None, :] * v_cache_stride_d +
                        (offs_blk_sz[:, None] + blk_start_n) * v_cache_stride_blk_sz)
        V_Cache = tl.load(v_cache_ptr + offs_v_cache, mask=mask_v_cache, other=0.0)
        
        acc, m_i, l_i = _dlm_fd_inner_kv_cache(Q, K_Cache, V_Cache, acc, m_i, l_i, softmax_scale, BLOCK_M, BLOCK_SIZE)
    
    # Then compute the main attention
    block_mask = tl.where(blk_start_m * BLOCK_M < q_seq_len, 1, 0)
    for blk_start_n in tl.range(0, block_mask * (blk_start_m + 1) * BLOCK_M, BLOCK_N, loop_unroll_factor=NUM_UNROLL_Q):
        blk_start_n = tl.multiple_of(blk_start_n, BLOCK_N)
        
        mask_k = (offs_n[None, :] + blk_start_n) < q_seq_len
        offs_k = (offs_n[None, :] + blk_start_n) * k_stride_n + kv_head_id * k_stride_nh + offs_hdim[:, None] * k_stride_d
        K = tl.load(k_ptr + offs_k, mask=mask_k, other=0.0)

        mask_v = (offs_n[:, None] + blk_start_n) < q_seq_len
        offs_v = (offs_n[:, None] + blk_start_n) * v_stride_n + kv_head_id * v_stride_nh + offs_hdim[None, :] * v_stride_d
        V = tl.load(v_ptr + offs_v, mask=mask_v, other=0.0)

        mask_m = mask_q & mask_k
        offs_mask = offs_m[:, None] * mask_stride_m + (offs_n[None, :] + blk_start_n) * mask_stride_n
        Mask = tl.load(mask_ptr + offs_mask, mask=mask_m, other=False)
        
        acc, m_i, l_i = _dlm_fd_inner_self(Q, K, V, Mask, acc, m_i, l_i, softmax_scale, BLOCK_M, BLOCK_N)
    
    # Store the output
    acc = ((1 / (l_i[:, None])) * acc).to(o_ptr.type.element_ty)
    mask_o = offs_m[:, None] < q_seq_len
    offs_o = (offs_m[:, None] + q_start_id) * o_stride_m + head_id * o_stride_nh + offs_hdim[None, :] * o_stride_d
    tl.store(o_ptr + offs_o, acc, mask=mask_o)


def diffusion_lm_flash_decoding(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, 
                                k_cache: torch.Tensor, v_cache: torch.Tensor, block_tables: torch.Tensor, 
                                cu_seqlens_q: torch.Tensor, seq_lens: torch.Tensor, total_lens: torch.Tensor, ctx_lens: torch.Tensor,
                                max_total_len: int | None = None, max_seq_len: int | None = None, 
                                diffusion_block_size: int = 32):
    '''
        FIXME
        q: [TotalInputLength, NumHeads, HeadDim]
        k: [TotalInputLength, NumHeads, HeadDim]
        v: [TotalInputLength, NumHeads, HeadDim]
        mask: [TotalInputLength, TotalInputLength]
        k_cache: [NumBlocks, NumHeads, HeadDim // x, BlockSize, x]
        v_cache: [NumBlocks, NumHeads, HeadDim, BlockSize]
        block_tables: [NumSeqs, MaxSeqNumBlocks] # NumSeqs == BatchSize
        ...
    '''
    is_pow_of_2 = lambda x: (x & (x - 1)) == 0 and x > 0
    assert k_cache.shape[-2] == v_cache.shape[-1], "BLOCK_SIZE between k_cache and v_cache must match"
    assert k.shape == v.shape, "k, v must have the same shape"
    assert k.shape[1] == k_cache.shape[1] == v_cache.shape[1], "Number of heads must match"
    assert q.shape[1] % k.shape[1] == 0, "Number of heads in q must be a multiple of the number of heads in k and v"
    assert k_cache.shape[-3] * k_cache.shape[-1] == v_cache.shape[-2] == q.shape[-1], "Head dimension must match"
    assert is_pow_of_2(q.shape[-1]) and is_pow_of_2(k_cache.shape[-3] * k_cache.shape[-1]), \
        "Head dimension must be a multiple of 2 for triton kernel compatibility"
    assert len(seq_lens) == len(ctx_lens) == len(total_lens) == len(cu_seqlens_q) - 1 == len(block_tables), \
        "Number of sequences must match across all inputs"

    BLOCK_SIZE = k_cache.shape[-2]  # BLOCK_SIZE or PAGE_SIZE of paged kv cache
    NUM_SEQS = len(ctx_lens)
    NUM_HEADS = q.shape[1]
    o = torch.empty_like(q).to(q.device).to(q.dtype)
    x = k_cache.shape[-1]
    max_seq_len = max_seq_len if max_seq_len is not None else max(seq_lens)
    max_total_len = max_total_len if max_total_len is not None else max(total_lens)
    softmax_scale = 1.0 / (k.shape[-1] ** 0.5)
    
    KV_HEAD_GROUP_SIZE = q.shape[1] // k.shape[1]
    HEAD_DIM = q.shape[-1]
    BLOCK_M = BLOCK_N = diffusion_block_size * 2
    GRID = (NUM_SEQS, NUM_HEADS, triton.cdiv(max_seq_len, BLOCK_M))
    
    dlm_flash_decoding_kernel[GRID](
        q, k, v, o, mask, softmax_scale, k_cache, v_cache, block_tables,
        cu_seqlens_q, total_lens, ctx_lens,
        *q.stride(), *k.stride(), *v.stride(), *o.stride(), *mask.stride(),
        *k_cache.stride(), *v_cache.stride(), *block_tables.stride(),
        cu_seqlens_q.stride(0), total_lens.stride(0), ctx_lens.stride(0),
        NUM_HEADS=NUM_HEADS, HEAD_DIM=HEAD_DIM, 
        KV_HEAD_GROUP_SIZE=KV_HEAD_GROUP_SIZE,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, x=x,
        BLOCK_SIZE=BLOCK_SIZE, 
        NUM_UNROLL_CACHE=4, NUM_UNROLL_Q=1
    )
    return o