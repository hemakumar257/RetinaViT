import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Dict, List, Any, Optional, Tuple

class ViTExplainer:
    """
    Explainability tools for RetinaViT models.
    Supports Attention Rollout and Grad-CAM for Transformers.
    """
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()
        self.attentions = []

    def _get_attention_hook(self):
        def hook(module, input, output):
            # Input to attn_drop is the attention matrix after softmax
            self.attentions.append(input[0].detach().cpu())
        return hook

    def compute_attention_rollout(self, images: torch.Tensor, discard_ratio: float = 0.9) -> np.ndarray:
        """
        Computes Attention Rollout by aggregating attention across all layers.
        """
        self.attentions = []
        hooks = []
        
        # Register hooks on all Attention drops
        for name, module in self.model.named_modules():
            if 'attn_drop' in name:
                hooks.append(module.register_forward_hook(self._get_attention_hook()))
        
        with torch.no_grad():
            _ = self.model(images.to(self.device))
            
        for h in hooks:
            h.remove()
            
        if not self.attentions:
            return None
            
        # Rollout computation
        # self.attentions is a list of [B, H, N, N]
        batch_size = self.attentions[0].shape[0]
        num_tokens = self.attentions[0].shape[-1]
        
        result = torch.eye(num_tokens).unsqueeze(0).repeat(batch_size, 1, 1) # [B, N, N]
        
        for attention in self.attentions:
            # Average over heads
            attn_heads_avg = attention.mean(dim=1) # [B, N, N]
            
            # Discard lowest weights
            if discard_ratio > 0:
                flat = attn_heads_avg.view(batch_size, -1)
                _, indices = flat.topk(int(num_tokens * num_tokens * (1 - discard_ratio)), dim=-1, largest=True)
                mask = torch.zeros_like(flat)
                mask.scatter_(1, indices, 1)
                attn_heads_avg = (attn_heads_avg * mask.view(batch_size, num_tokens, num_tokens))
                
            # Add Identity to account for residual connections
            I = torch.eye(num_tokens).unsqueeze(0)
            a = (attn_heads_avg + I) / 2
            a = a / a.sum(dim=-1).unsqueeze(-1)
            
            result = torch.matmul(a, result)
            
        # Extract attention from CLS token (token 0) to all other tokens
        # result: [B, N, N], we want result[:, 0, 1:]
        mask = result[:, 0, 1:]
        
        # Reshape to grid
        grid_size = int(np.sqrt(mask.shape[-1]))
        mask = mask.view(batch_size, grid_size, grid_size).numpy()
        
        # Normalize
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        return mask

    def compute_grad_cam(self, images: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Grad-CAM for Transformers focusing on the last attention block.
        """
        self.model.zero_grad()
        images = images.to(self.device)
        images.requires_grad = True
        
        # We need to capture the attention map and its gradients
        self.attentions = []
        self.gradients = []
        
        def save_grad(grad):
            self.gradients.append(grad.detach().cpu())
            
        def hook(module, input, output):
            self.attentions.append(input[0].detach().cpu())
            input[0].register_hook(save_grad)
            
        # Use the last block's attn_drop
        target_layer = None
        for name, module in self.model.named_modules():
            if 'blocks' in name and 'attn_drop' in name:
                target_layer = module
                
        if target_layer is None:
            return None
            
        h = target_layer.register_forward_hook(hook)
        
        outputs = self.model(images)
        logits = outputs if not isinstance(outputs, dict) else outputs["logits"]
        
        if target_class is None:
            target_class = logits.argmax(dim=1)
            
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, target_class.view(-1, 1), 1.0)
        
        logits.backward(gradient=one_hot)
        h.remove()
        
        # Grad-CAM computation
        # attn: [B, H, N, N], grads: [B, H, N, N]
        attn = self.attentions[0]
        grads = self.gradients[0]
        
        # Average over token dimension to get importance per head?
        # Standard Grad-CAM: weights = grads.mean(dim=(heads, token_q, token_k))
        # For Transformers, we often look at CLS token [:, :, 0, 1:]
        weights = grads[:, :, 0, 1:].mean(dim=-1, keepdim=True) # [B, H, 1]
        cam = (weights * attn[:, :, 0, 1:]).sum(dim=1) # [B, N-1]
        
        cam = F.relu(cam)
        cam = cam.view(images.shape[0], int(np.sqrt(cam.shape[-1])), -1).detach().cpu().numpy()
        
        # Normalize
        for i in range(len(cam)):
            cam[i] = (cam[i] - cam[i].min()) / (cam[i].max() - cam[i].min() + 1e-8)
            
        return cam

def visualize_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Overlays a heatmap on an image.
    image: [H, W, 3] uint8
    heatmap: [H, W] float 0-1
    """
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return overlay
