import torch

from d2f_vllm.layers.attention.attention_v2_profile import load_kvcache_kernel_kv_both

if __name__ == "__main__":
    torch.random.manual_seed(114514)
    
    seq_lens = torch.tensor([32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 
                32, 32, 32, 32, 32, 32, 32, 32, 32, 32]).to(torch.int32).to("cuda")
    ctx_lens = torch.tensor([119, 110,  81, 114, 112, 119, 110,  81, 114, 112, 
                119, 110,  81, 114, 112, 120, 111,  82, 115, 113]).to(torch.int32).to("cuda")
    total_lens = seq_lens + ctx_lens
    block_tables = torch.tensor(
        [[ 0],[ 1],[ 2],[ 3],[ 4],[ 5],[ 6], 
         [ 7],[ 8],[ 9],[10],[11],[12],[13],
         [14],[15],[16],[17],[18],[19]]).to(torch.int32).to("cuda")
    
    kv_cache_shape = (307, 256, 4, 128)
    k_cache = torch.randn(kv_cache_shape).to("cuda")
    v_cache = torch.randn_like(k_cache).to("cuda")

    kv_new_shape = (sum(seq_lens), 4, 128)
    k_new = torch.randn(kv_new_shape).to("cuda")
    v_new = torch.randn_like(k_new).to("cuda")
    
    kv_out_shape = (sum(total_lens), 4, 128)
    k_output = torch.empty(kv_out_shape).to("cuda")
    v_output = torch.empty_like(k_output).to("cuda")
    
    build_up_cu_seqlens = lambda x: torch.tensor([0] + list(torch.cumsum(x, dim=0).cpu().numpy())).to(torch.int32).to("cuda")
    cu_seqlens_q = build_up_cu_seqlens(seq_lens)
    cu_seqlens_k = build_up_cu_seqlens(total_lens)

    GRID = (block_tables.shape[0], block_tables.shape[1], k_new.shape[1])
    load_kvcache_kernel_kv_both[GRID](
        k_cache, v_cache,
        k_new, v_new,
        block_tables,
        k_output, v_output,
        seq_lens, ctx_lens,
        cu_seqlens_q, cu_seqlens_k, 
        *k_cache.stride(),
        *k_new.stride(),
        *block_tables.stride(),
        *k_output.stride(),
        ctx_lens.stride(0),
        seq_lens.stride(0),
        cu_seqlens_q.stride(0),
        cu_seqlens_k.stride(0),
        HEAD_DIM=128,
        MEM_BLOCK_SIZE=256,
        DIFFUSION_BLOCK_SIZE=32
    )
    
    k_output
    v_output