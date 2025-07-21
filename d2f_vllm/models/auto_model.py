from d2f_vllm.config import Config
from d2f_vllm.utils.loader import load_model
from d2f_vllm.models.dream import DreamForDLM
from d2f_vllm.models.qwen3 import Qwen3ForCausalLM

class AutoModelLM:
    MODEL_MAPPING = {
        "qwen3": Qwen3ForCausalLM,
        "dream": DreamForDLM
    }
    
    @classmethod
    def from_pretrained(self, config: Config):
        model = self.MODEL_MAPPING[config.model_name](config.hf_config)
        return load_model(model, config.model)