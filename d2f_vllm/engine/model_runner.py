import pickle
import torch
import torch.distributed as dist

from typing import List
from abc import ABC, abstractmethod
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from d2f_vllm.config import Config
from d2f_vllm.engine.sequence import SequenceForCausalLM, SequenceForDiffusionLM, SequenceBase
from d2f_vllm.models.auto_model import AutoModelLM
from d2f_vllm.layers.sampler import AutoSampler
from d2f_vllm.utils.context import (
    set_context_causal_lm, 
    get_context_causal_lm, 
    reset_context_causal_lm,
    set_context_diffusion_lm,
    get_context_diffusion_lm,
    reset_context_diffusion_lm
)


class ModelRunnerBase(ABC):
    """Base class for model runners supporting different model types."""
    def __init__(self, config: Config, rank: int, event: Event | List[Event]):
        self.config = config
        self.model_type = config.model_type
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        # Initialize model, sampler, and kv cache
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        self.model = AutoModelLM.from_config(config)
        self.sampler = AutoSampler.from_config(config)
        self.warmup_model()
        self.allocate_kv_cache() # NOCHANGE
        if not self.enforce_eager:
            self.capture_cudagraph()
        
        # Allocate shared memory for inter-process communication
        # NOCHANGE
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)
        if self.world_size > 1:
            if rank == 0:
                try:
                    shm = SharedMemory(name="d2f_vllm")
                    shm.close()
                    shm.unlink()
                    print("Removed existing shared memory segment.")
                except FileNotFoundError:
                    print("Shared memory segment does not exist, creating a new one.")
                self.shm = SharedMemory(name="d2f_vllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="d2f_vllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and not self.rank
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    @abstractmethod
    def warmup_model(self):
        """Model-specific warmup logic."""
        pass

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        
        if hasattr(hf_config, 'head_dim'):
            head_dim = hf_config.head_dim
        elif hasattr(hf_config, 'hidden_size') and hasattr(hf_config, 'num_attention_heads'):
            head_dim = hf_config.hidden_size // hf_config.num_attention_heads
        else:
            raise AttributeError(f"Cannot determine head_dim from config: {type(hf_config)}")
        
        block_bytes = (2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * 
                       head_dim * hf_config.torch_dtype.itemsize)
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - 
                                        used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        # [kv_separated, layer_id, block_id, block_size(segmented seq_len), head, head_dim]
        self.kv_cache = torch.zeros(
            2, hf_config.num_hidden_layers, config.num_kvcache_blocks, 
            self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: List[SequenceBase]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    @abstractmethod
    def prepare_prefill(self, seqs: List[SequenceBase]):
        """Model-specific prefill preparation."""
        pass

    @abstractmethod
    def prepare_decode(self, seqs: List[SequenceBase]):
        """Model-specific decode preparation."""
        pass

    def prepare_sample(self, seqs: List[SequenceBase]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @abstractmethod
    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        """Model-specific forward pass."""
        pass

    @abstractmethod
    def run(self, seqs: List[SequenceBase], is_prefill: bool) -> List[int]:
        """Main inference pipeline."""
        pass

    @abstractmethod
    @torch.inference_mode()
    def capture_cudagraph(self):
        """Model-specific CUDA graph capture."""
        pass


class ModelRunnerForCausalLM(ModelRunnerBase):
    """Model runner for Causal Language Models."""
    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        test_input_ids = [0] * max_model_len
        seqs = [SequenceForCausalLM(test_input_ids) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def prepare_prefill(self, seqs: List[SequenceForCausalLM]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context_causal_lm(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: List[SequenceForCausalLM]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq))
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context_causal_lm(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context_causal_lm()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            for k, v in graph_vars.items():
                if k != "outputs":
                    v.zero_()
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: List[SequenceForCausalLM], is_prefill: bool) -> List[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context_causal_lm()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context_causal_lm(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context_causal_lm()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )


class ModelRunnerForDiffusionLM(ModelRunnerBase):
    """Model runner for Diffusion Language Models. TODO: Implement DLM-specific logic."""
    def __init__(self, config: Config, rank: int, event: Event | List[Event]):
        super().__init__(config, rank, event)
        self.diffusion_block_size = config.diffusion_block_size
        self.mask_token_id = config.mask_token_id
            
    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        test_input_ids = [0] * max_model_len
        seqs = [SequenceForDiffusionLM(test_input_ids, config=self.config) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def prepare_prefill(self, seqs: List[SequenceForDiffusionLM]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seq.next_diffusion_step(is_prefill=True)
            
            total_seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, total_seqlen)))
            
            seqlen_q = total_seqlen - seq.num_cached_tokens
            seqlen_k = total_seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            
            if not seq.block_table:
                continue
            for i in range(seq.num_cached_blocks, seq.num_prompt_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_prompt_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_prompt_num_tokens 
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context_diffusion_lm(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables, seqs)
        return input_ids, positions

    def prepare_decode(self, seqs: List[SequenceForDiffusionLM]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        seq_split = []
        seq_id_to_queue_id = {}
        # print("="*20)
        for seq_idx_in_queue, seq in enumerate(seqs): 
            seq_id = seq.seq_id
            seq_id_to_queue_id[seq_id] = seq_idx_in_queue
            # print(seq_id)
            seq.next_diffusion_step()
            temp_input_ids, temp_positions, temp_context_len = seq.diffusion_decoding_inputs()
            seq_split.append(len(temp_input_ids))
            input_ids.extend(temp_input_ids)
            positions.extend(temp_positions)
            context_lens.append(temp_context_len)
            mem_block_to_diffusion_blocks_map = seq.mem_block_to_diffusion_blocks_map
            for mem_block_idx in range(seq.num_cached_blocks, seq.num_prompt_blocks):
                left_idx = mem_block_idx * seq.block_size
                end_idx = left_idx + seq.block_size
                cur_map = mem_block_to_diffusion_blocks_map[mem_block_idx]
                context_len = context_lens[seq_id_to_queue_id[seq_id]]
                is_last_block = False
                meet_active_block = False
                while left_idx < end_idx and not is_last_block and not meet_active_block:
                    local_left_idx = lambda: left_idx % seq.block_size
                    diffusion_block = seq.diffusion_blocks[cur_map[local_left_idx()]]
                    if cur_map[local_left_idx()] == seq.num_diffusion_blocks - 1:
                        is_last_block = True
                    get_step = lambda diff_blk: (
                        diff_blk.left_length
                        if diff_blk.left_length + local_left_idx() <= seq.block_size 
                        else seq.block_size - local_left_idx()
                    )
                    if diffusion_block.is_in_cache:
                        step = get_step(diffusion_block)
                        diffusion_block.cursor += step
                        left_idx += step
                    elif diffusion_block.is_to_cache:
                        step = get_step(diffusion_block)
                        cur_diffusion_block_start = diffusion_block.cursor
                        cur_diffusion_block_end = diffusion_block.cursor = diffusion_block.cursor + step
                        left_idx += step
                        mem_block_start = seq.block_table[mem_block_idx] * self.block_size + context_len % seq.block_size
                        context_len += step
                        slot_mapping.extend(list(range(mem_block_start + cur_diffusion_block_start,
                                                       mem_block_start + cur_diffusion_block_end)))
                    elif diffusion_block.is_active:
                        meet_active_block = True
                if meet_active_block:
                    break
        
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context_diffusion_lm(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables, seqs=seqs, seq_split=seq_split)
        return input_ids, positions

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context_diffusion_lm()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            for k, v in graph_vars.items():
                if k != "outputs":
                    v.zero_()
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: List[SequenceBase], is_prefill: bool) -> List[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        sample_output = self.sampler(logits, temperatures) if self.rank == 0 else None
        reset_context_diffusion_lm()
        return sample_output

    @torch.inference_mode()
    def capture_cudagraph(self):
        # TODO: Implement DLM-specific CUDA graph capture
        pass
    

class AutoModelRunner:
    @classmethod
    def from_config(cls, config: Config, rank: int, event: Event | List[Event]):
        """Factory method to create a model runner based on the model type."""
        if config.model_type == "causal_lm":
            return ModelRunnerForCausalLM(config, rank, event)
        elif config.model_type == "diffusion_lm":
            return ModelRunnerForDiffusionLM(config, rank, event)
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")