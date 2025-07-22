import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LoRALinear(nn.Module):
    """Linear layer with LoRA support."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 r: int = 0, lora_alpha: float = 1.0, lora_dropout: float = 0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0
        
        # Base linear layer
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        
        # LoRA parameters
        if r > 0:
            self.lora_A = nn.Parameter(torch.empty(r, in_features))
            self.lora_B = nn.Parameter(torch.empty(out_features, r))
            self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()
            self.merged = False
        else:
            self.lora_A = None
            self.lora_B = None
            self.merged = True
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        if hasattr(self, 'lora_A') and self.lora_A is not None:
            nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
            nn.init.zeros_(self.lora_B)
    
    def merge_lora(self):
        """Merge LoRA weights into base weight for efficient inference."""
        if not self.merged and self.r > 0:
            self.weight.data += self.scaling * torch.mm(self.lora_B, self.lora_A)
            self.merged = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged or self.r == 0:
            return F.linear(x, self.weight, self.bias)
        else:
            base_out = F.linear(x, self.weight, self.bias)
            lora_out = F.linear(self.lora_dropout(x), self.lora_A.T)
            lora_out = F.linear(lora_out, self.lora_B.T)
            return base_out + lora_out * self.scaling


class ColumnParallelLoRALinear(LoRALinear):
    """Column parallel linear layer with LoRA support."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 gather_output: bool = True, r: int = 0, lora_alpha: float = 1.0):
        super().__init__(in_features, out_features, bias, r, lora_alpha)
        self.gather_output = gather_output


class RowParallelLoRALinear(LoRALinear):
    """Row parallel linear layer with LoRA support."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 input_is_parallel: bool = False, r: int = 0, lora_alpha: float = 1.0):
        super().__init__(in_features, out_features, bias, r, lora_alpha)
        self.input_is_parallel = input_is_parallel


def convert_linear_to_lora(module: nn.Module, target_modules: list = None, r: int = 16, 
                          lora_alpha: float = 32.0) -> nn.Module:
    """Convert regular Linear layers to LoRA-enabled layers."""
    if target_modules is None:
        target_modules = ["Linear", "ColumnParallelLinear", "RowParallelLinear"]
    
    for name, child in module.named_children():
        if child.__class__.__name__ in target_modules:
            if hasattr(child, 'in_features') and hasattr(child, 'out_features'):
                # Create LoRA layer
                lora_layer = LoRALinear(
                    child.in_features, 
                    child.out_features, 
                    bias=child.bias is not None,
                    r=r, 
                    lora_alpha=lora_alpha
                )
                # Copy weights
                lora_layer.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    lora_layer.bias.data.copy_(child.bias.data)
                
                setattr(module, name, lora_layer)
        else:
            convert_linear_to_lora(child, target_modules, r, lora_alpha)
    
    return module
