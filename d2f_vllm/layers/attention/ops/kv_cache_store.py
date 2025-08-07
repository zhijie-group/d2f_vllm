import torch
import triton 
import triton.language as tl


@triton.jit
def store_kvcache_kernel_causal_lm(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr
):
    idx = tl.program_id(0)
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    slot = tl.load(slot_mapping_ptr + idx)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)
    

@triton.jit
def store_kvcache_kernel_diffusion_lm(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr
):
    token_idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + token_idx)
    if slot < 0:
        return
    key_offsets = token_idx * key_stride + tl.arange(0, D)
    value_offsets = token_idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


@triton.jit
def store_kvcache_kernel_diffusion_lm_distinct(
    k_ptr, v_ptr, k_cache_ptr, v_cache_ptr, slot_mapping_ptr,
    k_stride, v_stride,  
    k_cache_stride_nblks, k_cache_stride_h, k_cache_stride_dx, k_cache_stride_blk_sz, k_cache_stride_x,
    v_cache_stride_nblks, v_cache_stride_h, v_cache_stride_d, v_cache_stride_blk_sz,
    nheads, hdim, blk_sz,
    x: tl.constexpr, D: tl.constexpr
):
    # Translated from vLLM's CUDA kernel (
    # Referencing https://github.com/vllm-project/vllm/blob/main/csrc/cache_kernels.cu#L212
    # and https://github.com/vllm-project/vllm/blob/main/csrc/cache_kernels.cu#L415
    token_idx = tl.program_id(0)
    slot_idx = tl.load(slot_mapping_ptr + token_idx)
    if slot_idx < 0:
        return
    
    blk_idx = slot_idx // blk_sz
    off_blk = slot_idx % blk_sz
    
    offs_d = tl.arange(0, D)
    offs_k = token_idx * k_stride + offs_d
    offs_v = token_idx * v_stride + offs_d
    k = tl.load(k_ptr + offs_k)
    v = tl.load(v_ptr + offs_v)

    h_ids = offs_d // hdim
    h_offs = offs_d % hdim
    x_ids = h_offs // x
    x_offs = h_offs % x
    
    k_cache_offs = (blk_idx * k_cache_stride_nblks + h_ids * k_cache_stride_h +
                    x_ids * k_cache_stride_dx + off_blk * k_cache_stride_blk_sz + 
                    x_offs * k_cache_stride_x)
    v_cache_offs = (blk_idx * v_cache_stride_nblks + h_ids * v_cache_stride_h +
                    h_offs * v_cache_stride_d + off_blk * v_cache_stride_blk_sz)
    
    tl.store(k_cache_ptr + k_cache_offs, k)
    tl.store(v_cache_ptr + v_cache_offs, v)
    

def store_kvcache_distinct_layout(key: torch.Tensor, value: torch.Tensor, 
                                  k_cache: torch.Tensor, v_cache: torch.Tensor, 
                                  slot_mapping: torch.Tensor, model_type: str = 'causal_lm') -> None:
    
    if model_type == 'causal_lm':
        # k_cache: [num_blks, blk_sz, h, hdim]
        # v_cache: [num_blks, blk_sz, h, hdim]
        N, num_heads, head_dim = key.shape
        D = num_heads * head_dim
        assert key.stride(-1) == 1 and value.stride(-1) == 1
        assert key.stride(1) == head_dim and value.stride(1) == head_dim
        assert k_cache.stride(1) == D and v_cache.stride(1) == D
        assert N == slot_mapping.numel()
        store_kvcache_kernel_causal_lm[(N,)](
            key, key.stride(0),
            value, value.stride(0),
            k_cache, v_cache, slot_mapping, D
        )
    else:
        # TODO: implement diffusion lm kv cache store
        # k_cache: [num_blks, h, hdim // x, blk_sz, x]
        # v_cache: [num_blks, h, hdim, blk_sz]
        NBlks, NHeads, HDim_x, Blk_sz, x = k_cache.shape
        HDim = HDim_x * x
        N = key.shape[0]
        assert HDim == key.shape[-1] and NHeads == key.shape[1]
        assert N == slot_mapping.numel()
        
        GRID = (N, )
        store_kvcache_kernel_diffusion_lm_distinct[GRID](
            key, value,
            k_cache, v_cache,
            slot_mapping,
            key.stride(0), value.stride(0), 
            *k_cache.stride(), *v_cache.stride(),
            NHeads, HDim, Blk_sz,
            x, HDim * NHeads
        )


def store_kvcache_unified(key: torch.Tensor, value: torch.Tensor, 
                          k_cache: torch.Tensor, v_cache: torch.Tensor, 
                          slot_mapping: torch.Tensor, model_type: str = 'causal_lm') -> None:
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert N == slot_mapping.numel()

    if model_type == 'causal_lm':
        store_kvcache_kernel_causal_lm[(N,)](
            key, key.stride(0),
            value, value.stride(0),
            k_cache, v_cache, slot_mapping, D
        )
    elif model_type == 'diffusion_lm':
        store_kvcache_kernel_diffusion_lm[(N,)](
            key, key.stride(0),
            value, value.stride(0),
            k_cache, v_cache, slot_mapping, D
        )