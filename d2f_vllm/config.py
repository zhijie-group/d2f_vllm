import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    lora_path: str = ""
    model_name: str = "dream"
    model_type: str = "diffusion_lm" # "causal_lm" or "diffusion_lm"
    mask_token_id: int = 151666
    diffusion_block_size: int = 32
    
    accept_threshold: float = 0.95
    complete_threshold: float = 0.9
    add_new_block_threshold: float = 0.1
    
    use_lora: bool = False
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        
        # LoRA validation
        if self.use_lora:
            if not self.lora_path:
                raise ValueError("lora_path must be provided when use_lora is True")
            if not os.path.exists(self.lora_path):
                print(f"Warning: LoRA path {self.lora_path} does not exist")
        
        self.hf_config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
