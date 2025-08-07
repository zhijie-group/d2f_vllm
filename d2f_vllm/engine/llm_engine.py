import atexit

import torch.multiprocessing as mp

from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from typing import List

from d2f_vllm.config import Config
from d2f_vllm.sampling_params import SamplingParams
from d2f_vllm.engine.sequence import SequenceForCausalLM, SequenceForDiffusionLM
from d2f_vllm.engine.scheduler import AutoScheduler, SchedulerBase
from d2f_vllm.engine.model_runner import AutoModelRunner


class LLMEngine:
    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        self.config = config = Config(model, **config_kwargs)
        self.engine_type = config.model_type
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=AutoModelRunner.from_config, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = AutoModelRunner.from_config(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True, trust_remote_code=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler: SchedulerBase = AutoScheduler.from_config(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | List[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
            
        if self.engine_type == "causal_lm":
            seq = SequenceForCausalLM(prompt, sampling_params)
        elif self.engine_type == "diffusion_lm":
            seq = SequenceForDiffusionLM(prompt, sampling_params, config=self.config)
        else:
            raise ValueError(f"Unsupported engine type: {self.engine_type}")
        
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        sample_output = self.model_runner.call("run_verbose", seqs, is_prefill)
        self.scheduler.postprocess(seqs, sample_output)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        if self.engine_type == "causal_lm":
            num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        else:
            num_tokens = sum(seq.diffusion_num_tokens for seq in seqs) if is_prefill else sum(seq.new_tokens for seq in seqs)
        return outputs, num_tokens, is_prefill

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: List[str] | List[List[int]],
        sampling_params: SamplingParams | List[SamplingParams],
        use_tqdm: bool = True,
    ) -> List[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        n_steps = 0
        while not self.is_finished():
            t = perf_counter()
            n_steps += 1
            output, num_tokens, is_prefill = self.step()
            if use_tqdm:
                if is_prefill:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        print(f"Finished in {n_steps} steps, prefill throughput: {prefill_throughput:.2f} tok/s, decode throughput: {decode_throughput:.2f} tok/s")
        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
