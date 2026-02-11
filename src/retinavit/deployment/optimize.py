import torch
import torch.nn as nn
import torch.quantization
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List
import copy
import os

def quantize_model_int8(model: nn.Module, dataloader: Optional[DataLoader] = None, cfg: Optional[Dict[str, Any]] = None) -> nn.Module:
    """
    Applies post-training dynamic quantization to the model (Linear components).
    For static quantization, a dataloader is required for calibration.
    """
    model.eval()
    
    # Dynamic quantization for Linear and RNN layers is straightforward
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {nn.Linear}, 
        dtype=torch.qint8
    )
    
    return quantized_model

def prune_attention_heads(model: nn.Module, pruning_ratio: float, cfg: Optional[Dict[str, Any]] = None) -> nn.Module:
    """
    Structured pruning for Transformer attention heads based on L1 norm.
    Simplification: for this demo, we zero out heads or return a pruned version if using a specific library.
    """
    # In a real implementation with timm, we would identify heads in Attention blocks
    # and mask out the lowest importance heads.
    # For now, we simulate by applying global unstructured pruning to Linear layers if requested,
    # or identify heads if structural info is available.
    
    import torch.nn.utils.prune as prune
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
            prune.remove(module, 'weight') # making it permanent
            
    return model

def export_torchscript(model: nn.Module, out_path: str, input_size: List[int] = [1, 3, 224, 224], cfg: Optional[Dict[str, Any]] = None):
    """
    Exports the model to TorchScript for mobile and optimized inference.
    """
    model.eval()
    example_input = torch.randn(*input_size).to(next(model.parameters()).device)
    
    # Trace the model
    try:
        traced_model = torch.jit.trace(model, example_input)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        traced_model.save(out_path)
        print(f"Model exported successfully to {out_path}")
    except Exception as e:
        print(f"Failed to export via trace, trying script... Error: {e}")
        scripted_model = torch.jit.script(model)
        scripted_model.save(out_path)
        print(f"Model exported via script to {out_path}")

if __name__ == "__main__":
    # Simplified smoke test
    class SimpleViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = nn.Linear(3*16*16, 128)
            self.transformer = nn.Linear(128, 128)
            self.head = nn.Linear(128, 5)
        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = self.patch_embed(x)
            x = self.transformer(x)
            return self.head(x)

    model = SimpleViT()
    q_model = quantize_model_int8(model)
    print("Quantization success")
    
    p_model = prune_attention_heads(model, 0.3)
    print("Pruning success")
    
    export_torchscript(model, "checkpoints/deployment/test_model.pt", input_size=[1, 3, 16, 16])
