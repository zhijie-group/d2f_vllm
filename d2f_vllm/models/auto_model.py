from d2f_vllm.config import Config
from d2f_vllm.utils.loader import load_model
from d2f_vllm.models.dream import DreamForDiffusionLM
from d2f_vllm.models.qwen3 import Qwen3ForCausalLM


class AutoModelLM:
    MODEL_MAPPING = {
        "qwen3": Qwen3ForCausalLM,
        "dream": DreamForDiffusionLM
    }
    @classmethod
    def from_pretrained(cls, config: Config):
        model = cls.MODEL_MAPPING[config.model_name](config.hf_config)
        return load_model(model, config)