import torch

from copy import copy
from enum import Enum, auto
from itertools import count
from typing import List, Tuple, Any
from dataclasses import dataclass

from d2f_vllm.config import Config
from d2f_vllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class SequenceBase:
    block_size = 256
    counter = count()
    
    def __init__(self, token_ids: List[int], sampling_params: SamplingParams = SamplingParams()):
        self.seq_id = next(SequenceBase.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self) -> int:
        return self.num_tokens

    def __getitem__(self, key) -> int:
        return self.token_ids[key]

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self) -> int:
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self) -> List[int]:
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def num_cached_blocks(self) -> int:
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self) -> int:
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self) -> int:
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i) -> List[int]:
        assert 0 <= i < self.num_blocks
        return self.token_ids[i * self.block_size: (i + 1) * self.block_size]

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1


class SequenceForCausalLM(SequenceBase):
    """Standard sequence implementation for Causal Language Models."""
    
    def __init__(self, token_ids: List[int], sampling_params = SamplingParams()):
        super().__init__(token_ids, sampling_params)
        
    def __repr__(self) -> str:
        return (f"SequenceForCausalLM(block_size={self.block_size}, counter={self.counter}, "
                f"seq_id={self.seq_id}, status={self.status.name}, num_tokens={self.num_tokens}, "
                f"num_prompt_tokens={self.num_prompt_tokens}, num_cached_tokens={self.num_cached_tokens}, "
                f"temperature={self.temperature}, max_tokens={self.max_tokens}, ignore_eos={self.ignore_eos})")
        
    def __getstate__(self) -> Tuple[int, int, int, List[int], int]:
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state: Tuple[int, int, int, List[int], int]) -> None:
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
            
    @property
    def completion_token_ids(self) -> List[int]:
        return self.token_ids[self.num_prompt_tokens:]


class DiffusionBlockStatus(Enum):
    ACTIVE = auto()
    TO_CACHE = auto()
    IN_CACHE = auto()


@dataclass
class DiffusionBlock:
    block_id: int = 0
    status: DiffusionBlockStatus = DiffusionBlockStatus.ACTIVE
    
    global_start_id: int = 0
    global_end_id: int | None = None
    cursor: int = 0
    
    mask_token_id: int = 151666
    size: int = 32
    is_prompt: bool = False
    
    accept_threshold: float = 0.95 # Threshold to accept a token in the diffusion block
    add_new_block_threshold: float = 0.1 # Threshold to add a new block
    complete_threshold: float = 0.9 # Can only be cached when the current diffusion block is completed
    
    seq: "SequenceForDiffusionLM" = None # Reference to the sequence this block belongs to
    pre_block: "DiffusionBlock" = None # Create prefix linked list of diffusion blocks
    suf_block: "DiffusionBlock" = None # Create suffix linked list of diffusion blocks
    
    def __post_init__(self):
        self.global_end_id = self.global_start_id + self.size

    def __getitem__(self, key: int) -> int:
        return self.seq[self.global_start_id + key]
    
    def __len__(self) -> int:
        return self.size
    
    @property
    def current_complete_ratio(self) -> float:
        return sum([token_id != self.mask_token_id for token_id in self.token_ids]) / self.size

    @property
    def available_to_cache(self) -> bool:
        return self.current_complete_ratio == 1.0

    @property
    def is_active(self) -> bool:
        return self.status == DiffusionBlockStatus.ACTIVE
    
    @property
    def is_in_cache(self) -> bool:
        return self.status == DiffusionBlockStatus.IN_CACHE

    @property
    def is_to_cache(self) -> bool:
        return self.status == DiffusionBlockStatus.TO_CACHE
    
    @property
    def pre_block_complete(self) -> bool:
        return self.pre_block.current_complete_ratio >= self.complete_threshold if self.pre_block is not None else True
    
    @property
    def add_new_block(self) -> bool:
        return self.current_complete_ratio >= self.add_new_block_threshold
    
    @property
    def token_ids(self) -> torch.Tensor:
        if self.seq is not None:
            return self.seq.token_ids[self.global_start_id:self.global_end_id]
        else:
            raise RuntimeError("Sequence is not set for the diffusion block.")
        
    @property
    def local_mask_tokens(self) -> List[bool]:
        return [token_id == self.seq.mask_token_id for token_id in self.token_ids]
    
    @property
    def local_mask_token_ids(self) -> List[int]:
        return [idx for idx, mask_token in enumerate(self.local_mask_tokens) if mask_token]

    @property
    def global_mask_token_ids(self) -> List[int]:
        offset = self.global_start_id
        in_cache_blocks = list(range(sum(self.seq.in_cache_blocks)))
        offset -= sum(self.seq.diffusion_blocks[block_id].size for block_id in in_cache_blocks)
        return [mask_token_id + offset for mask_token_id in self.local_mask_token_ids]
    
    @property
    def left_length(self) -> int:
        return self.size - self.cursor
        
    def to_cache(self) -> None:
        if self.available_to_cache and not self.is_in_cache:
            self.status = DiffusionBlockStatus.TO_CACHE
    
    def in_cache(self) -> None:
        if self.is_to_cache:
            self.status = DiffusionBlockStatus.IN_CACHE
        
    def modify_token(self, local_token_id: int, modified_to: int) -> None:
        self.seq.token_ids[local_token_id + self.global_start_id] = modified_to.item()
        self.seq.new_tokens += 1
    

class SequenceForDiffusionLM(SequenceBase):
    """Sequence implementation for Diffusion Language Models."""

    def __init__(self, token_ids: List[int], 
                 sampling_params = SamplingParams(),
                 config: Config = None):
        super().__init__(token_ids, sampling_params)
        self.config = config
        self.eos_token_id = config.eos
        self.max_model_len = config.max_model_len
        self.mask_token_id = config.mask_token_id
        self.diffusion_block_size = config.diffusion_block_size
        self.block_mask = None
        self.meet_eos = False
        self.diffusion_blocks: List[DiffusionBlock] = []
    
    def __getstate__(self):
        diffusion_blocks_state = []
        for block in self.diffusion_blocks:
            diffusion_blocks_state.append({
                'block_id': block.block_id,
                'status': block.status,
                'global_start_id': block.global_start_id,
                'global_end_id': block.global_end_id,
                'mask_token_id': block.mask_token_id,
                'size': block.size,
                'is_prompt': block.is_prompt,
                'accept_threshold': block.accept_threshold,
                'add_new_block_threshold': block.add_new_block_threshold,
                'complete_threshold': block.complete_threshold,
            })

        state = {
            "seq_id": self.seq_id,
            "status": self.status,
            "token_ids": self.token_ids,
            "last_token": self.last_token,
            "num_tokens": self.num_tokens,
            "num_prompt_tokens": self.num_prompt_tokens,
            "num_cached_tokens": self.num_cached_tokens,
            "block_table": self.block_table,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "ignore_eos": self.ignore_eos,
            "config": self.config,
            "eos_token_id": self.eos_token_id,
            "max_model_len": self.max_model_len,
            "mask_token_id": self.mask_token_id,
            "diffusion_block_size": self.diffusion_block_size,
            "diffusion_blocks_state": diffusion_blocks_state,
            "input_token_ids": getattr(self, "input_token_ids", []),
            "input_num_tokens": getattr(self, "input_num_tokens", 0),
            "input_num_prompt_tokens": getattr(self, "input_num_prompt_tokens", 0),
            "new_tokens": getattr(self, "new_tokens", 0),
            "meet_eos": self.meet_eos,
            "block_mask": self.block_mask,
        }
        return state

    def __setstate__(self, state):
        self.seq_id = state["seq_id"]
        self.status = state["status"]
        self.token_ids = state["token_ids"]
        self.last_token = state["last_token"]
        self.num_tokens = state["num_tokens"]
        self.num_prompt_tokens = state["num_prompt_tokens"]
        self.num_cached_tokens = state["num_cached_tokens"]
        self.block_table = state["block_table"]
        self.temperature = state["temperature"]
        self.max_tokens = state["max_tokens"]
        self.ignore_eos = state["ignore_eos"]
        self.meet_eos = state["meet_eos"]

        self.config = state["config"]
        self.eos_token_id = state["eos_token_id"]
        self.max_model_len = state["max_model_len"]
        self.mask_token_id = state["mask_token_id"]
        self.diffusion_block_size = state["diffusion_block_size"]

        self.input_token_ids = state.get("input_token_ids", [])
        self.input_num_tokens = state.get("input_num_tokens", 0)
        self.input_num_prompt_tokens = state.get("input_num_prompt_tokens", 0)
        self.new_tokens = state.get("new_tokens", 0)
        self.block_mask = state.get("block_mask", None)

        self.diffusion_blocks = []
        pre_block = None
        for block_state in state["diffusion_blocks_state"]:
            block = DiffusionBlock(
                block_id=block_state["block_id"],
                status=block_state["status"],
                global_start_id=block_state["global_start_id"],
                global_end_id=block_state["global_end_id"],
                mask_token_id=block_state["mask_token_id"],
                size=block_state["size"],
                is_prompt=block_state["is_prompt"],
                accept_threshold=block_state.get("accept_threshold", 0.95),
                add_new_block_threshold=block_state.get("add_new_block_threshold", 0.1),
                complete_threshold=block_state.get("complete_threshold", 0.9),
                seq=self,
                pre_block=pre_block,
            )
            if pre_block is not None:
                pre_block.suf_block = block
            self.diffusion_blocks.append(block)
            pre_block = block
    
    def __repr__(self) -> str:
        return (f"SequenceForDiffusionLM(block_size={self.block_size}, counter={self.counter}, "
                f"seq_id={self.seq_id}, status={self.status.name}, num_tokens={self.num_tokens}, "
                f"num_prompt_tokens={self.num_prompt_tokens}, num_cached_tokens={self.num_cached_tokens}, "
                f"temperature={self.temperature}, max_tokens={self.max_tokens}, ignore_eos={self.ignore_eos}, "
                f"diffusion_block_size={self.diffusion_block_size}, current_block_mask={self.current_block_mask.shape}, "
                f"input_token_ids={self.input_token_ids}, input_num_tokens={self.input_num_tokens})")
    
    @property
    def completion_token_ids(self) -> List[int]:
        return self.token_ids[self.input_num_prompt_tokens:]
    
    @property
    def active_blocks(self) -> List[bool]:
        return [block.is_active for block in self.diffusion_blocks]
    
    @property
    def to_cache_blocks(self) -> List[bool]:
        return [block.is_to_cache for block in self.diffusion_blocks]

    @property
    def in_cache_blocks(self) -> List[bool]:
        return [block.is_in_cache for block in self.diffusion_blocks]
    
    @property
    def current_block_mask(self) -> torch.Tensor:
        return self.block_mask[..., :self.num_tokens, :self.num_tokens]
    
    @property
    def num_prompt_blocks(self) -> int:
        return (self.input_num_prompt_tokens + self.block_size - 1) // self.block_size
    
    @property
    def last_block_prompt_num_tokens(self) -> int:
        return self.input_num_prompt_tokens - (self.num_prompt_blocks - 1) * self.block_size
    
    @property
    def updated_kv_cache_blocks(self) -> List[int]:
        return [idx for idx, caching in enumerate(self.caching_blocks) if caching]

    @property
    def caching_blocks(self) -> List[bool]:
        return [to_cache or in_cache for to_cache, in_cache in zip(self.to_cache_blocks, self.in_cache_blocks)]
    
    @property
    def cached_block_ids(self) -> List[int]:
        return [idx for idx, in_cache in enumerate(self.in_cache_blocks) if in_cache]

    @property
    def mask_tokens(self) -> List[bool]:
        return [token_id == self.mask_token_id for token_id in self.token_ids]
    
    @property
    def cached_last_token_id(self) -> int:
        cached_num_tokens = 0
        for block_id in self.updated_kv_cache_blocks:
            block = self.diffusion_blocks[block_id]
            cached_num_tokens += block.size
        return cached_num_tokens - 1
    
    @property
    def cached_num_tokens(self) -> int:
        return self.cached_last_token_id + 1
    
    @property
    def diffusion_num_tokens(self) -> int:
        return sum(self.mask_tokens)
    
    @property
    def mem_block_to_diffusion_blocks_map(self) -> List[List[int]]:
        mapping = []
        for block_id in range(self.num_blocks):
            window_start = block_id * self.block_size
            window_length = self.block_size if block_id < self.num_blocks - 1 else self.last_block_num_tokens
            mapping.append([self.token_to_diffusion_block_id(token_id) for token_id in range(window_start, window_start + window_length)]) # build up token-wise mapping
        return mapping
    
    def token_to_diffusion_block_id(self, token_id: int) -> int:
        if token_id < self.input_num_tokens:
            return 0
        else:
            return (token_id - self.input_num_tokens) // self.diffusion_block_size + 1
    
    @property
    def num_diffusion_blocks(self) -> int:
        return len(self.diffusion_blocks)
    
    def diffusion_decoding_inputs(self) -> Tuple[List[int], List[int], int]:
        to_cache_and_active_blocks = self.diffusion_blocks[self.cached_block_ids[-1] + 1:]
        assert len(to_cache_and_active_blocks) == sum(self.active_blocks) + sum(self.to_cache_blocks)
        
        input_tokens = []
        positions = []
        context_len = sum(self.diffusion_blocks[block_id].size for block_id in self.cached_block_ids)
        temp_context_len = context_len
        for block in to_cache_and_active_blocks:
            input_tokens.extend(block.token_ids)
            positions.extend([token_id + temp_context_len for token_id in range(block.size)])
            temp_context_len += block.size
            
        return input_tokens, positions, context_len

    def reset_new_tokens(self) -> None:
        self.new_tokens = 0
    
    def post_process(self) -> None:
        for block_id, block in enumerate(self.diffusion_blocks):
            block.cursor = 0
            previous_caching_blocks = self.caching_blocks[:block_id]
            if block.is_to_cache:
                block.in_cache()
            elif sum(block.local_mask_tokens) == 0 and sum(previous_caching_blocks) == len(previous_caching_blocks):
                block.to_cache()
            else:
                break
        
    def generate_block_mask(self) -> None:
        max_model_len = self.config.max_model_len
        mask_shape = [1, 1, max_model_len, max_model_len]
        block_wise_causal_mask = torch.zeros(mask_shape, dtype=torch.bool, device="cuda")
        block_wise_causal_mask[..., :self.input_num_tokens, :self.input_num_tokens] = True
        num_diffusion_blocks = (max_model_len - self.input_num_tokens + self.diffusion_block_size - 1) // self.diffusion_block_size
        for block_id in range(num_diffusion_blocks):
            start_h = self.input_num_tokens + block_id * self.diffusion_block_size
            end_h = start_h + self.diffusion_block_size
            start_w = 0
            end_w = end_h
            block_wise_causal_mask[..., start_h:end_h, start_w:end_w] = True
        self.block_mask = block_wise_causal_mask.clone()

    def next_diffusion_step(self, is_prefill: bool = False) -> None:
        if is_prefill:
            # Take a snapshot of the original input state
            self.input_token_ids = self.token_ids.copy()
            self.input_num_tokens = self.num_tokens
            self.input_num_prompt_tokens = self.num_prompt_tokens
            self.num_prompt_tokens += self.diffusion_block_size
            
            self.diffusion_blocks.append(
                DiffusionBlock(
                    block_id=len(self.diffusion_blocks),
                    status=DiffusionBlockStatus.TO_CACHE,
                    global_start_id=0,
                    mask_token_id=self.mask_token_id,
                    size=len(self.input_token_ids),
                    accept_threshold=self.config.accept_threshold,
                    add_new_block_threshold=self.config.add_new_block_threshold,
                    complete_threshold=self.config.complete_threshold,
                    is_prompt=True,
                    seq=self
                )
            )
        
        if self.diffusion_blocks[-1].add_new_block and not self.meet_eos:
            added_num_tokens = (
                self.diffusion_block_size 
                if self.num_tokens + self.diffusion_block_size <= self.max_model_len 
                else self.max_model_len - self.num_tokens
            )
            
            diffusion_seq = [self.mask_token_id] * added_num_tokens
            current_diffusion_block = DiffusionBlock(
                block_id=len(self.diffusion_blocks),
                status=DiffusionBlockStatus.ACTIVE,
                global_start_id=self.num_tokens,
                mask_token_id=self.mask_token_id,
                size=added_num_tokens,
                accept_threshold=self.config.accept_threshold,
                add_new_block_threshold=self.config.add_new_block_threshold,
                complete_threshold=self.config.complete_threshold,
                seq=self,
                pre_block=self.diffusion_blocks[-1] if self.diffusion_blocks else None
            )
            
            self.diffusion_blocks[-1].suf_block = current_diffusion_block
            self.token_ids += diffusion_seq
            self.num_tokens += added_num_tokens
            self.diffusion_blocks.append(current_diffusion_block)

        if self.block_mask is None and is_prefill:
            self.generate_block_mask()