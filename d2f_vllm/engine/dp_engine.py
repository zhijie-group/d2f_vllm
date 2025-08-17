import atexit
import multiprocessing as mp
import os
import torch
from multiprocessing.connection import wait as mp_wait

from typing import List, Any

from d2f_vllm.config import Config
from d2f_vllm.engine.llm_engine import LLMEngine
from d2f_vllm.sampling_params import SamplingParams


def _dp_child_entry(config: Config, dp_idx: int, local_devices: list[int], conn):
    """Child process entry point: create an LLMEngine for this DP rank and serve RPC via Pipe."""
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in local_devices)
        cfg = Config(
            model=config.model,
            lora_path=config.lora_path,
            model_name=config.model_name,
            model_type=config.model_type,
            mask_token_id=config.mask_token_id,
            diffusion_block_size=config.diffusion_block_size,
            accept_threshold=config.accept_threshold,
            complete_threshold=config.complete_threshold,
            add_new_block_threshold=config.add_new_block_threshold,
            use_lora=config.use_lora,
            max_num_batched_tokens=config.max_num_batched_tokens,
            max_num_seqs=config.max_num_seqs,
            max_model_len=config.max_model_len,
            gpu_memory_utilization=config.gpu_memory_utilization,
            data_parallel_size=1,
            tensor_parallel_size=config.tensor_parallel_size,
            master_addr=config.master_addr,
            master_port=int(config.master_port) + dp_idx,
            shm_name=f"{config.shm_name}_{dp_idx}",
            enforce_eager=config.enforce_eager,
            kvcache_block_size=config.kvcache_block_size,
            num_kvcache_blocks=config.num_kvcache_blocks,
            k_cache_hdim_split_factor_x=config.k_cache_hdim_split_factor_x,
            kv_cache_layout=config.kv_cache_layout,
        )
        setattr(cfg, "device_start", 0)

        engine = LLMEngine(cfg.model, **{k: getattr(cfg, k) for k in cfg.__dataclass_fields__.keys() if k != "model"})

        while True:
            msg = conn.recv()
            if not msg:
                continue
            cmd, *args = msg
            if cmd == "generate":
                plist, sp_arg, use_tqdm = args
                out = engine.generate(plist, sp_arg, use_tqdm)
                conn.send(("ok", out))
                continue
            if cmd == "exit":
                engine.exit()
                conn.send(("ok", None))
                break
            elif cmd == "add_request":
                prompt, sp = args
                engine.add_request(prompt, sp)
                conn.send(("ok", None))
            elif cmd == "step":
                out = engine.step()
                conn.send(("ok", out))
            elif cmd == "is_finished":
                conn.send(("ok", engine.is_finished()))
            else:
                conn.send(("err", f"unknown_cmd:{cmd}"))
    except Exception as e:
        try:
            conn.send(("err", repr(e)))
        except Exception:
            pass


class DPEngine:
    """Data-parallel wrapper that runs N independent TP groups as child processes and aggregates results."""
    def __init__(self, model, **kwargs):
        config_fields = {f for f in Config.__dataclass_fields__.keys()}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        self.config = cfg = Config(model, **config_kwargs)
        self.dp_size = cfg.data_parallel_size
        assert self.dp_size > 1, "Use LLMEngine directly when data_parallel_size == 1"

        ctx = mp.get_context("spawn")
        self.conns: List[Any] = []
        self.ps: List[mp.Process] = []
        # Topology check and mapping
        base_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if base_visible:
            vis = [int(x) for x in base_visible.split(',') if x.strip() != '']
        else:
            vis = list(range(torch.cuda.device_count()))

        need_gpus = self.dp_size * cfg.tensor_parallel_size
        assert len(vis) >= need_gpus, f"Require {need_gpus} GPUs (dp={self.dp_size}, tp={cfg.tensor_parallel_size}), visible {len(vis)}"

        # Optional overrides: kwargs['device_ids'] or env D2F_DEVICE_MAP
        override = None
        if 'device_ids' in kwargs and kwargs['device_ids']:
            override = list(kwargs['device_ids'])
        elif os.environ.get('D2F_DEVICE_MAP'):
            override = [int(x) for x in os.environ['D2F_DEVICE_MAP'].split(',') if x.strip() != '']
        if override is not None:
            assert len(override) >= need_gpus, f"device_ids length {len(override)} < required {need_gpus}"
            # All override devices must be in visible list
            assert all(d in vis for d in override[:need_gpus]), "device_ids must be subset of CUDA_VISIBLE_DEVICES"
            plan = override[:need_gpus]
        else:
            plan = vis[:need_gpus]

        tp = cfg.tensor_parallel_size
        for dp_idx in range(self.dp_size):
            local_devices = plan[dp_idx*tp:(dp_idx+1)*tp]
            parent_conn, child_conn = ctx.Pipe()
            p = ctx.Process(target=_dp_child_entry, args=(cfg, dp_idx, local_devices, child_conn))
            p.start()
            self.ps.append(p)
            self.conns.append(parent_conn)
        self._rr = 0  # round-robin pointer
        atexit.register(self.exit)

    def _ask(self, replica: int, cmd: str, *args):
        conn = self.conns[replica]
        conn.send((cmd, *args))
        tag, payload = conn.recv()
        if tag == "ok":
            return payload
        raise RuntimeError(f"DP child #{replica} error: {payload}")

    def exit(self):
        for i, p in enumerate(self.ps):
            if p.is_alive():
                try:
                    self._ask(i, "exit")
                except Exception:
                    pass
                p.join(timeout=5)

    def add_request(self, prompt: str | List[int], sampling_params: SamplingParams):
        target = self._rr
        self._rr = (self._rr + 1) % self.dp_size
        self._ask(target, "add_request", prompt, sampling_params)

    def step(self):
        all_outputs = []
        total_tokens = 0
        any_prefill = False
        merged_diff_steps = {}
        for i in range(self.dp_size):
            done = self._ask(i, "is_finished")
            if done:
                continue
            outputs, num_tokens, is_prefill, n_diff_steps = self._ask(i, "step")
            if outputs:
                all_outputs.extend(outputs)
            total_tokens += num_tokens
            any_prefill = any_prefill or is_prefill
            if n_diff_steps:
                merged_diff_steps.update(n_diff_steps)
        return all_outputs, total_tokens, any_prefill, merged_diff_steps

    def is_finished(self):
        return all(self._ask(i, "is_finished") for i in range(self.dp_size))

    def generate(self, prompts: List[str] | List[List[int]], sampling_params: SamplingParams | List[SamplingParams], use_tqdm: bool = True):
        n = len(prompts)
        step_size = n // self.dp_size
        results = []
        pending = {}
        conn_to_idx = {}
        for i in range(self.dp_size):
            s = i * step_size
            e = (i + 1) * step_size if i < self.dp_size - 1 else n
            if isinstance(sampling_params, list):
                if len(sampling_params) == n:
                    sp_arg = sampling_params[s:e]
                elif len(sampling_params) == self.dp_size:
                    sp_arg = sampling_params[i]
                else:
                    sp_arg = sampling_params[0]
            else:
                sp_arg = sampling_params
            conn = self.conns[i]
            conn.send(("generate", prompts[s:e], sp_arg, use_tqdm))
            pending[i] = True
            conn_to_idx[conn] = i
        while pending:
            ready = mp_wait([self.conns[i] for i in pending.keys()])
            for conn in ready:
                tag, payload = conn.recv()
                if tag == "ok":
                    results.extend(payload)
                else:
                    raise RuntimeError(f"DP child #{conn_to_idx[conn]} error: {payload}")
                del pending[conn_to_idx[conn]]
        return results
