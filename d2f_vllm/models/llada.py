# d2f_vllm/models/llada.py

import os
import torch
import torch.nn as nn
import torch.distributed as dist

from d2f_vllm.layers.activation import SiluAndMul
from d2f_vllm.layers.attention.attention_v4 import Attention
from d2f_vllm.layers.layernorm import RMSNorm
from d2f_vllm.layers.linear import RowParallelLinear, ColumnParallelLinear
from d2f_vllm.layers.rotary_embedding import get_rope
from d2f_vllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from d2f_vllm.models.config.llada.configuration_llada import LLaDAConfig


if os.environ.get("TRITON_INTERPRET", None) == "1":
    torch._dynamo.reset()
    torch._dynamo.config.suppress_errors = True
    torch.backends.optimized_mode = False


class LLaDARMSNorm(RMSNorm):
    """Llama-style RMSNorm."""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__(hidden_size, eps)


class LLaDAAttention(nn.Module):
    """LLaDA attention mechanism, adapted for vLLM."""
    def __init__(
        self,
        config: LLaDAConfig
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        
        # GQA/MQA support
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        
        self.head_dim = self.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        # Use separate ColumnParallelLinear for Q, K, V as in the original LLaDA/Llama
        self.q_proj = ColumnParallelLinear(
            self.hidden_size,
            self.total_num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = ColumnParallelLinear(
            self.hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = ColumnParallelLinear(
            self.hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
        )
        
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            "llama", # Use "llama" attention type for causal masking
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v, mask)
        output = self.o_proj(o)
        return output


class LLaDAMLP(nn.Module):
    """LLaDA MLP (SwiGLU), adapted for vLLM."""
    def __init__(
        self,
        config: LLaDAConfig,
    ) -> None:
        super().__init__()
        self.gate_proj = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )
        self.up_proj = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )
        assert config.hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The SiluAndMul kernel expects a concatenated input [gate, up]
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        x = self.act_fn(torch.cat([gate, up], dim=-1))
        x = self.down_proj(x)
        return x


class LLaDADecoderLayer(nn.Module):
    """LLaDA transformer decoder layer, Llama-style."""
    def __init__(
        self,
        config: LLaDAConfig,
    ) -> None:
        super().__init__()
        self.self_attn = LLaDAAttention(config)
        self.mlp = LLaDAMLP(config)
        self.input_layernorm = LLaDARMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LLaDARMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Llama-style pre-norm and residual connection
        
        # Self Attention block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states, mask)
        hidden_states = residual + hidden_states

        # MLP block
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class LLaDAModel(nn.Module):
    """LLaDA model body."""
    def __init__(
        self,
        config: LLaDAConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LLaDADecoderLayer(config) 
                                     for _ in range(config.num_hidden_layers)])
        self.norm = LLaDARMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states, mask)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class LLaDAForCausalLM(nn.Module):
    """Full LLaDA model for Causal Language Modeling."""
    packed_modules_mapping = {}
    
    def __init__(
        self,
        config: LLaDAConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = LLaDAModel(config)
        # Note: In Llama, the LM head is not parallelized in the same way, but using ParallelLMHead
        # is the standard and correct way to do it in a tensor-parallel framework like vLLM.
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, mask)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        return logits