import os
import json
import torch
import torch.nn as nn

from tqdm import tqdm
from glob import glob
from safetensors import safe_open
from d2f_vllm.config import Config


def load_lora_config(lora_path: str) -> dict:
    """Load LoRA configuration from adapter_config.json."""
    config_path = os.path.join(lora_path, "adapter_config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}


def enable_lora_for_model(model: nn.Module, lora_config: dict):
    """Enable LoRA for existing linear layers in the model."""
    r = lora_config.get('r', 16)
    lora_alpha = lora_config.get('lora_alpha', 32.0)
    
    for module in model.modules():
        if hasattr(module, '__init_lora__'):
            module.__init_lora__(r, lora_alpha)
    return model


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, config: Config):
    """Load model weights and optionally LoRA weights."""
    # Enable LoRA for linear layers if LoRA is enabled
    if config.use_lora and config.lora_path:
        lora_config = load_lora_config(config.lora_path)
        if lora_config:
            print(f"Found LoRA config: r={lora_config.get('r', 16)}, alpha={lora_config.get('lora_alpha', 32.0)}")
            model = enable_lora_for_model(model, lora_config)
        else:
            print("No adapter_config.json found, using default LoRA parameters")
            model = enable_lora_for_model(model, {'r': 16, 'lora_alpha': 32.0})
    
    # Load base model weights
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in tqdm(glob(os.path.join(config.model, "*.safetensors")), desc="Loading base model"):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
    
    # Load LoRA weights if enabled
    if config.use_lora and config.lora_path:
        if os.path.exists(config.lora_path):
            print(f"Loading LoRA weights from {config.lora_path}")
            model = load_lora_weights(model, config.lora_path)
        else:
            print(f"Warning: LoRA path {config.lora_path} does not exist, skipping LoRA loading")
    
    return model


def load_lora_weights(model: nn.Module, lora_path: str):
    """Load LoRA weights into LoRA-enabled layers."""
    try:
        # Load LoRA config for additional info
        lora_config = load_lora_config(lora_path)
        target_modules = lora_config.get('target_modules', [])
        
        lora_weights = {}
        
        # Load all LoRA weights
        for file in tqdm(glob(os.path.join(lora_path, "*.safetensors")), desc="Loading LoRA"):
            with safe_open(file, "pt", "cpu") as f:
                for weight_name in f.keys():
                    lora_weights[weight_name] = f.get_tensor(weight_name)
        
        # Apply LoRA weights to model
        applied_count = 0
        for name, module in model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                # Check if this module should have LoRA applied
                should_apply = True
                if target_modules:
                    should_apply = any(target in name for target in target_modules)
                
                if not should_apply:
                    continue
                
                # Look for corresponding LoRA weights with various naming patterns
                base_patterns = [
                    name,
                    f"base_model.model.{name}",
                    f"model.{name}",
                ]
                
                found_a = found_b = None
                for base_name in base_patterns:
                    lora_a_keys = [
                        f"{base_name}.lora_A.weight",
                        f"{base_name}.lora_A",
                        f"{base_name}.lora_A.default.weight"
                    ]
                    lora_b_keys = [
                        f"{base_name}.lora_B.weight", 
                        f"{base_name}.lora_B",
                        f"{base_name}.lora_B.default.weight"
                    ]
                    
                    for key in lora_a_keys:
                        if key in lora_weights:
                            found_a = lora_weights[key]
                            break
                    for key in lora_b_keys:
                        if key in lora_weights:
                            found_b = lora_weights[key]
                            break
                    
                    if found_a is not None and found_b is not None:
                        break
                
                if found_a is not None and found_b is not None:
                    # Handle tensor parallel sharding if needed
                    if hasattr(module, 'tp_size') and module.tp_size > 1:
                        if hasattr(module, 'tp_dim') and module.tp_dim == 0:  # Column parallel
                            shard_size = found_b.size(0) // module.tp_size
                            start_idx = module.tp_rank * shard_size
                            found_b = found_b[start_idx:start_idx + shard_size]
                        elif hasattr(module, 'tp_dim') and module.tp_dim == 1:  # Row parallel
                            shard_size = found_a.size(1) // module.tp_size
                            start_idx = module.tp_rank * shard_size
                            found_a = found_a[:, start_idx:start_idx + shard_size]
                    
                    try:
                        module.lora_A.data.copy_(found_a)
                        module.lora_B.data.copy_(found_b)
                        applied_count += 1
                    except Exception as e:
                        print(f"Failed to load LoRA weights for {name}: {e}")
        
        # Merge LoRA weights for efficient inference
        for module in model.modules():
            if hasattr(module, 'merge_lora'):
                module.merge_lora()
        
        print(f"LoRA weights applied to {applied_count} layers and merged")
        
    except Exception as e:
        print(f"Error loading LoRA weights: {e}")
        print("Continuing with base model only")
    
    return model
