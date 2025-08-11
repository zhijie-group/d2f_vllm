# d2f_vllm/models/config/llada/configuration_llada.py

from dataclasses import dataclass

@dataclass
class LLaDAConfig:
    """
    Configuration for the LLaDA model, adapted for vLLM.
    """
    model_type: str = "llada"

    # Core model architecture
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32  # For GQA, this would be smaller
    hidden_act: str = "silu"

    # Vocabulary and embeddings
    vocab_size: int = 32000
    tie_word_embeddings: bool = False

    # Positional embeddings and attention
    rope_theta: float = 10000.0
    rope_scaling: dict | None = None
    max_position_embeddings: int = 2048
    use_cache: bool = True
    
    # Normalization
    rms_norm_eps: float = 1e-6

    # Bias
    attention_bias: bool = False # Llama-2 style doesn't use bias in QKV
    
    def __post_init__(self):
        # Support for older configs that might use these names
        if hasattr(self, 'd_model'):
            self.hidden_size = self.d_model
        if hasattr(self, 'n_heads'):
            self.num_attention_heads = self.n_heads
        if hasattr(self, 'n_kv_heads') and self.n_kv_heads is not None:
            self.num_key_value_heads = self.n_kv_heads
        else:
            self.num_key_value_heads = self.num_attention_heads # Default to MHA if not specified
        if hasattr(self, 'n_layers'):
            self.num_hidden_layers = self.n_layers
        if hasattr(self, 'max_sequence_length'):
            self.max_position_embeddings = self.max_sequence_length

    # The following are properties to match HuggingFace's config API if needed
    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads