import torch

from copy import copy
from enum import Enum, auto
from itertools import count
from typing import List, Tuple
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
    def completion_token_ids(self) -> List[int]:
        return self.token_ids[self.num_prompt_tokens:]

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

    def __getstate__(self) -> Tuple[int, int, int, List[int], int]:
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state: Tuple[int, int, int, List[int], int]) -> None:
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]


class SequenceForCausalLM(SequenceBase):
    """Standard sequence implementation for Causal Language Models."""
    
    def __init__(self, token_ids: List[int], sampling_params = SamplingParams()):
        super().__init__(token_ids, sampling_params)
        
    def __repr__(self) -> str:
        return (f"SequenceForCausalLM(block_size={self.block_size}, counter={self.counter}, "
                f"seq_id={self.seq_id}, status={self.status.name}, num_tokens={self.num_tokens}, "
                f"num_prompt_tokens={self.num_prompt_tokens}, num_cached_tokens={self.num_cached_tokens}, "
                f"temperature={self.temperature}, max_tokens={self.max_tokens}, ignore_eos={self.ignore_eos})")


class DiffusionBlockStatus(Enum):
    ACTIVE = auto()
    TO_CACHE = auto()
    IN_CACHE = auto()


@dataclass
class DiffusionBlock:
    status: DiffusionBlockStatus = DiffusionBlockStatus.ACTIVE
    token_ids: torch.Tensor = None
    mask_token_id: int = 151666
    size: int = 32
    to_cache_threshold: float = 0.9
    is_prompt: bool = False
    is_completed: bool = False
    
    def __post_init__(self):
        pass

    @property
    def current_rate(self) -> float:
        return sum(torch.where(self.token_ids == self.mask_token_id)) / self.size
    
    @property
    def available_to_cache(self) -> bool:
        return self.current_rate >= self.to_cache_threshold

    @property
    def is_active(self) -> bool:
        return self.status == DiffusionBlockStatus.ACTIVE
    
    @property
    def is_in_cache(self) -> bool:
        return self.status == DiffusionBlockStatus.IN_CACHE

    @property
    def is_to_cache(self) -> bool:
        return self.status == DiffusionBlockStatus.TO_CACHE
    
    def to_cache(self) -> None:
        if self.available_to_cache:
            self.status = DiffusionBlockStatus.TO_CACHE
        else:
            raise RuntimeError("Cannot mark block for caching when it is not ready.")
    
    def in_cache(self) -> None:
        if self.is_to_cache:
            self.status = DiffusionBlockStatus.IN_CACHE
        elif self.is_in_cache:
            raise RuntimeError("Cannot cache a block that is already in cache.")
    

class SequenceForDiffusionLM(SequenceBase):
    """Sequence implementation for Diffusion Language Models."""
    
    def __init__(self, token_ids: List[int], 
                 sampling_params = SamplingParams(),
                 config: Config = None):
        super().__init__(token_ids, sampling_params)
        self.config = config
        self.max_model_len = config.max_model_len
        self.mask_token_id = config.mask_token_id
        self.diffusion_block_size = config.diffusion_block_size
        self.block_mask = None
        self.diffusion_blocks: List[DiffusionBlock] = []

    def __repr__(self) -> str:
        return (f"SequenceForDiffusionLM(block_size={self.block_size}, counter={self.counter}, "
                f"seq_id={self.seq_id}, status={self.status.name}, num_tokens={self.num_tokens}, "
                f"num_prompt_tokens={self.num_prompt_tokens}, num_cached_tokens={self.num_cached_tokens}, "
                f"temperature={self.temperature}, max_tokens={self.max_tokens}, ignore_eos={self.ignore_eos}, "
                f"diffusion_block_size={self.diffusion_block_size}, current_block_mask={self.current_block_mask.shape}, "
                f"input_token_ids={self.input_token_ids}, input_num_tokens={self.input_num_tokens})")
        
    def generate_block_mask(self) -> None:
        max_model_len = self.config.max_model_len
        mask_shape = [1, 1, max_model_len, max_model_len]
        block_wise_causal_mask = torch.zeros(mask_shape, dtype=torch.bool, device="cuda")
        block_wise_causal_mask[..., :self.input_num_tokens, :self.input_num_tokens] = True
        num_diffusion_blocks = (max_model_len - self.input_num_tokens +
                                self.diffusion_block_size - 1) // self.diffusion_block_size
        for block_id in range(num_diffusion_blocks):
            start_h = self.input_num_tokens + block_id * self.diffusion_block_size
            end_h = start_h + self.diffusion_block_size
            start_w = 0
            end_w = end_h
            block_wise_causal_mask[..., start_h:end_h, start_w:end_w] = True
        self.block_mask = block_wise_causal_mask.clone()
        
    @property
    def active_blocks(self) -> List[DiffusionBlock]:
        return [block.is_active for block in self.diffusion_blocks]
    
    @property
    def to_cache_blocks(self) -> List[DiffusionBlock]:
        return [block.is_to_cache for block in self.diffusion_blocks]

    @property
    def in_cache_blocks(self) -> List[DiffusionBlock]:
        return [block.is_in_cache for block in self.diffusion_blocks]

    @property
    def completed_blocks(self) -> List[DiffusionBlock]:
        return [block.is_completed for block in self.diffusion_blocks]
    
    @property
    def computing_blocks(self) -> List[DiffusionBlock]:
        return self.active_blocks or ((self.to_cache_blocks or self.in_cache_blocks) and not self.completed_blocks)
    
    @property
    def current_block_mask(self) -> torch.Tensor:
        return self.block_mask[..., :self.num_tokens, :self.num_tokens]
    
    @property
    def num_prompt_blocks(self) -> int:
        return (self.input_num_prompt_tokens + self.block_size - 1) // self.block_size
    
    @property
    def last_block_prompt_num_tokens(self) -> int:
        return self.input_num_prompt_tokens - (self.num_prompt_blocks - 1) * self.block_size

    def next_diffusion_step(self, is_prefill: bool = False) -> None:
        if is_prefill:
            # Take a snapshot of the original input state
            self.input_token_ids = self.token_ids.copy()
            self.input_num_tokens = self.num_tokens
            self.input_num_prompt_tokens = self.num_prompt_tokens
            self.num_prompt_tokens += self.diffusion_block_size
            self.diffusion_blocks.append(
                DiffusionBlock(
                    status=DiffusionBlockStatus.TO_CACHE,
                    token_ids=torch.tensor(self.input_token_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True),
                    mask_token_id=self.mask_token_id,
                    size=len(self.input_token_ids),
                    to_cache_threshold=self.config.to_cache_threshold,
                    is_prompt=True,
                )
            )
        added_num_tokens = (
            self.diffusion_block_size 
            if self.num_tokens + self.diffusion_block_size <= self.max_model_len 
            else self.max_model_len - self.num_tokens
        )
        diffusion_seq = [self.mask_token_id] * added_num_tokens
        current_diffusion_block = DiffusionBlock(
            status=DiffusionBlockStatus.ACTIVE,
            token_ids=torch.tensor(diffusion_seq, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True),
            mask_token_id=self.mask_token_id,
            size=added_num_tokens,
            to_cache_threshold=self.config.to_cache_threshold,
        )
        self.token_ids += diffusion_seq
        self.num_tokens += added_num_tokens
        self.diffusion_blocks.append(current_diffusion_block)

        if self.block_mask is None and is_prefill:
            self.generate_block_mask()