from copy import copy
from enum import Enum, auto
from itertools import count
from typing import List, Tuple

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
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos


class SequenceForCausalLM(SequenceBase):
    def __init__(self, token_ids: List[int], sampling_params = SamplingParams()):
        super().__init__(token_ids, sampling_params)
        
    def __repr__(self) -> str:
        return (f"Sequence(block_size={self.block_size}, counter={self.counter}, "
                f"seq_id={self.seq_id}, status={self.status.name}, num_tokens={self.num_tokens}, "
                f"num_prompt_tokens={self.num_prompt_tokens}, num_cached_tokens={self.num_cached_tokens}, "
                f"temperature={self.temperature}, max_tokens={self.max_tokens}, ignore_eos={self.ignore_eos})")

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


class SequenceForDiffusionLM(SequenceBase):
    def __init__(self, token_ids: List[int], 
                 sampling_params = SamplingParams(),
                 diffusion_block_size: int = 32):
        super().__init__(token_ids, sampling_params)
        self.diffusion_block_size = diffusion_block_size