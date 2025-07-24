from collections import deque
from typing import Tuple, List, Deque
from abc import ABC, abstractmethod

from d2f_vllm.config import Config
from d2f_vllm.engine.sequence import SequenceBase, SequenceStatus
from d2f_vllm.engine.block_manager import BlockManager


class SchedulerBase(ABC):
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: Deque[SequenceBase] = deque()
        self.running: Deque[SequenceBase] = deque()

    @abstractmethod
    def is_finished(self) -> bool:
        pass
    
    @abstractmethod
    def add(self, seq: SequenceBase) -> None:
        pass
    
    @abstractmethod
    def schedule(self) -> Tuple[List[SequenceBase], bool]:
        pass
    
    @abstractmethod
    def preempt(self, seq: SequenceBase) -> None:
        pass
    
    @abstractmethod
    def postprocess(self, seqs: List[SequenceBase], token_ids: List[int]) -> List[bool]:
        pass
    

class SchedulerForCausalLM(SchedulerBase):
    def __init__(self, config: Config):
        super().__init__(config)
        
    def is_finished(self) -> bool:
        return not self.waiting and not self.running

    def add(self, seq: SequenceBase) -> None:
        self.waiting.append(seq)

    def schedule(self) -> Tuple[List[SequenceBase], bool]:
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens \
                or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: SequenceBase) -> None:
        seq.status = SequenceStatus.WAITING
        self.block_manager.free(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: List[SequenceBase], token_ids: List[int]) -> List[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) \
                or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.free(seq)
                self.running.remove(seq)


# TODO
class SchedulerForDiffusionLM(SchedulerBase):
    def __init__(self, config: Config):
        super().__init__(config)
        self.diffusion_block_size = config.diffusion_block_size

    def is_finished(self) -> bool:
        return not self.waiting and not self.running

    def add(self, seq: SequenceBase) -> None:
        self.waiting.append(seq)
    
    def schedule(self):
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens \
                or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: SequenceBase) -> None:
        seq.status = SequenceStatus.WAITING
        self.block_manager.free(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: List[SequenceBase], token_ids: List[int]) -> List[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) \
                or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.free(seq)
                self.running.remove(seq)
    

class AutoScheduler:
    SCHEDULER_MAPPING = {
        "causal_lm": SchedulerForCausalLM,
        "diffusion_lm": SchedulerForDiffusionLM,
    }
    @classmethod
    def from_config(cls, config: Config):
        return cls.SCHEDULER_MAPPING[config.model_type](config)