import torch

from collections import deque
from typing import Tuple, List, Deque
from abc import ABC, abstractmethod

from d2f_vllm.config import Config
from d2f_vllm.engine.sequence import (
    SequenceBase, SequenceStatus, 
    SequenceForDiffusionLM, SequenceForCausalLM
)
from d2f_vllm.layers.sampler import SampleOutputForDiffusionLM
from d2f_vllm.engine.block_manager import AutoBlockManager


class SchedulerBase(ABC):
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = AutoBlockManager.from_config(config)
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
    def postprocess(self, seqs: List[SequenceBase], token_ids: List[int]):
        pass
    

class SchedulerForCausalLM(SchedulerBase):
    def __init__(self, config: Config):
        super().__init__(config)
        
    def is_finished(self) -> bool:
        return not self.waiting and not self.running

    def add(self, seq: SequenceForCausalLM) -> None:
        self.waiting.append(seq)

    def schedule(self) -> Tuple[List[SequenceForCausalLM], bool]:
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

    def preempt(self, seq: SequenceForCausalLM) -> None:
        seq.status = SequenceStatus.WAITING
        self.block_manager.free(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: List[SequenceForCausalLM], token_ids: List[int]) -> None:
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

    def add(self, seq: SequenceForDiffusionLM) -> None:
        self.waiting.append(seq)
    
    def schedule(self):
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) + seq.diffusion_block_size > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) + seq.diffusion_block_size - seq.num_cached_tokens
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

    def preempt(self, seq: SequenceForDiffusionLM) -> None:
        seq.status = SequenceStatus.WAITING
        self.block_manager.free(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: List[SequenceForDiffusionLM], sample_output: SampleOutputForDiffusionLM) -> None:
        for seq in seqs:
            seq.reset_new_tokens()
            seq_id = str(seq.seq_id)
            cur_true_local_ids_sub_map = sample_output.true_local_ids_map.get(seq_id, {})
            cur_accepted_ids_sub_map = sample_output.accepted_ids_map.get(seq_id, {})
            cur_sampled_tokens_sub_map = sample_output.sampled_tokens_map.get(seq_id, {})
            for block_id, accepted_ids in cur_accepted_ids_sub_map.items():
                if len(accepted_ids) > 0:
                    block = seq.diffusion_blocks[int(block_id)]
                    sampled_tokens = cur_sampled_tokens_sub_map.get(block_id, [])
                    true_local_ids = cur_true_local_ids_sub_map.get(block_id, [])

                    for true_local_id, accepted_id in zip(true_local_ids, accepted_ids):
                        block.modify_token(true_local_id, sampled_tokens[accepted_id])
                        if ((not seq.ignore_eos and sampled_tokens[accepted_id].item() == self.eos) 
                            or seq.num_completion_tokens == seq.max_tokens):
                            seq.status = SequenceStatus.FINISHED
                            self.block_manager.free(seq)
                            self.running.remove(seq)
            seq.post_process()

class AutoScheduler:
    SCHEDULER_MAPPING = {
        "causal_lm": SchedulerForCausalLM,
        "diffusion_lm": SchedulerForDiffusionLM,
    }
    @classmethod
    def from_config(cls, config: Config):
        return cls.SCHEDULER_MAPPING[config.model_type](config)