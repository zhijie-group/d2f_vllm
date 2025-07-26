import torch
import triton

import torch.nn as nn
import triton.language as tl

from typing import List
from functools import lru_cache
from einops import rearrange
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

from d2f_vllm.utils.context import (
    ContextForCausalLM, ContextForDiffusionLM, 
    get_context_causal_lm, get_context_diffusion_lm
)


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
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


def store_kvcache(
    key: torch.Tensor, value: torch.Tensor, 
    k_cache: torch.Tensor, v_cache: torch.Tensor, 
    slot_mapping: torch.Tensor, model_type: str = 'causal_lm') -> None:
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    
    N = slot_mapping.numel() if model_type == 'diffusion_lm' else N
    assert N == slot_mapping.numel()
    
    store_kvcache_kernel[(N,)](
        key, key.stride(0),
        value, value.stride(0),
        k_cache, v_cache, slot_mapping, D
    )
    

class Attention(nn.Module):
    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        model_type='causal_lm'
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.causal = model_type == 'causal_lm'
        self.model_type = model_type
        self.flex_attention = torch.compile(flex_attention, dynamic=False)
        self._block_mask_cache = {}
        
    @lru_cache(maxsize=32)
    def dllm_block_mask(self, block_mask: torch.Tensor, 
                        B: int, H: int, Q_LEN: int, KV_LEN: int, device: str):
        cache_key = (B, H, Q_LEN, KV_LEN, device)
        def _mask_mod(batch, head, token_q, token_kv):
            return block_mask[token_q, token_kv]
        if cache_key not in self._block_mask_cache:
            self._block_mask_cache[cache_key] = create_block_mask(
                _mask_mod, B, H, Q_LEN, KV_LEN, device=device
            )
        return self._block_mask_cache[cache_key]
   
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: List[torch.Tensor] | None = None) -> torch.Tensor:
        """
            TODO: 
            - Optimize Diffusion LM Attention !!!
            - Support Causal LM Attention through flex_attention to align with Diffusion LM
        """
        o: torch.Tensor
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        context: ContextForDiffusionLM = get_context_causal_lm() if self.model_type == 'causal_lm' else get_context_diffusion_lm()
        k_cache, v_cache = self.k_cache, self.v_cache
        
        if k_cache.numel() and v_cache.numel() and not(self.model_type == 'diffusion_lm' and context.slot_mapping.numel() == 0):
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping, self.model_type)
                
        if context.is_prefill: # prefill
            if context.block_tables is not None and self.model_type == 'causal_lm': # prefix cache
                k, v = k_cache, v_cache
            transpose_fn = lambda x: rearrange(x, 's h d -> h s d').unsqueeze(0)
            q, k, v = [transpose_fn(tensor) for tensor in (q, k, v)]
            B, H, S, _ = q.shape
            dllm_block_mask = self.dllm_block_mask(context.block_mask, B, H, S, S, str(q.device))
            o = self.flex_attention(q, k, v, block_mask=dllm_block_mask, enable_gqa=True).squeeze(0)
        else: # decode
            transpose_fn = lambda x: rearrange(x, 's h d -> h s d').unsqueeze(0)
            q = transpose_fn(q)
            k_list, v_list = [torch.split(tensor, context.seq_split, dim=0) for tensor in (k, v)]
            cat_k_list = []
            cat_v_list = []
            for seq_idx, (k, v) in enumerate(zip(k_list, v_list)):
                cur_context_len = context.context_lens[seq_idx]
                k_cache_temp, v_cache_temp = None, None
                for mem_block_idx in context.block_tables[seq_idx]:
                    k_mem_block, v_mem_block = k_cache[mem_block_idx], v_cache[mem_block_idx]
                    mem_block_size = k_cache.shape[1]
                    cur_window = mem_block_size if mem_block_size <= cur_context_len else cur_context_len % mem_block_size
                    cur_context_len = cur_context_len - cur_window
                    k_cache_temp = k_mem_block[:cur_window] if k_cache_temp is None else torch.cat((k_cache_temp, k_mem_block[:cur_window]), dim=0)
                    v_cache_temp = v_mem_block[:cur_window] if v_cache_temp is None else torch.cat((v_cache_temp, v_mem_block[:cur_window]), dim=0)
                cat_k_list.extend([k_cache_temp, k])
                cat_v_list.extend([v_cache_temp, v])
            k_cache, v_cache = transpose_fn(torch.cat(cat_k_list, dim=0)), transpose_fn(torch.cat(cat_v_list, dim=0))
            B, H, Sq, _ = q.shape
            B, H, Skv, _ = k_cache.shape
            dllm_block_mask = self.dllm_block_mask(context.block_mask, B, H, Sq, Skv, str(q.device))
            o = self.flex_attention(q, k_cache, v_cache, block_mask=dllm_block_mask, enable_gqa=True).squeeze(0)
            
        if self.model_type == 'causal_lm':
            o = o.view(-1, self.num_heads * self.head_dim)
        elif self.model_type == 'diffusion_lm':
            o = rearrange(o, 'h s d -> s (h d)')
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        return o