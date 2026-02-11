import torch
import torch.nn as nn
import timm
from typing import Dict, Any, Optional

class RetinaViT(nn.Module):
    """
    Standard Vision Transformer for Retinal Image Classification.
    Wraps timm ViT while exposing a clean interface.
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        model_cfg = cfg["model"]
        self.model_name = model_cfg["name"]
        
        # Determine model variant from name if not specified
        timm_variant = "vit_base_patch16_224" # Default fallback
        if "small" in self.model_name:
            timm_variant = "vit_small_patch16_224"
        elif "tiny" in self.model_name:
            timm_variant = "vit_tiny_patch16_224"
        elif "large" in self.model_name:
            timm_variant = "vit_large_patch16_224"
            
        self.backbone = timm.create_model(
            timm_variant,
            pretrained=model_cfg.get("pretrained", True),
            num_classes=model_cfg.get("num_classes", 5),
            img_size=model_cfg.get("image_size", [512, 512]),
            drop_rate=model_cfg.get("dropout", 0.1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

class QualityAwareRetinaViT(nn.Module):
    """
    Vision Transformer with Quality Conditioning.
    Supports quality token injection, attention modulation, and adaptive patch dropout.
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        model_cfg = cfg["model"]
        self.quality_cfg = model_cfg.get("quality_aware", {})
        
        # Initialize standard ViT backbone
        self.retina_vit = RetinaViT(cfg)
        self.backbone = self.retina_vit.backbone
        
        embed_dim = self.backbone.embed_dim
        
        # a) Quality Token Injection
        if self.quality_cfg.get("use_quality_token", False):
            # Discrete label embedding
            self.quality_label_embed = nn.Embedding(4, embed_dim // 2)
            # Continuous score MLP (assuming single quality score or small vector)
            self.quality_score_mlp = nn.Sequential(
                nn.Linear(1, embed_dim // 2),
                nn.GELU(),
                nn.Linear(embed_dim // 2, embed_dim // 2)
            )
            # Final token projection
            self.quality_token_proj = nn.Linear(embed_dim, embed_dim)
            self.quality_label_map = {"Good": 0, "Fair": 1, "Poor": 2, "Unknown": 3}
            
        # b) Attention Modulation Embedding
        if self.quality_cfg.get("use_attention_modulation", False):
            self.modulation_mlp = nn.Sequential(
                nn.Embedding(4, embed_dim),
                nn.Linear(embed_dim, embed_dim),
                nn.Sigmoid()
            )

    def forward(self, x: torch.Tensor, quality_info: Dict[str, Any]) -> torch.Tensor:
        # Extract quality labels and scores
        q_labels_str = quality_info.get("quality_label", ["Unknown"] * x.shape[0])
        q_indices = torch.tensor([self.quality_label_map.get(q, 3) for q in q_labels_str]).to(x.device)
        
        q_scores = quality_info.get("quality_score")
        if q_scores is None:
            q_scores = torch.zeros((x.shape[0], 1), device=x.device)
        elif isinstance(q_scores, list):
            q_scores = torch.tensor(q_scores, device=x.device).unsqueeze(-1).float()
            
        # Standard Patch Embedding
        x = self.backbone.patch_embed(x)
        
        # Class token
        cls_token = self.backbone.cls_token.expand(x.shape[0], -1, -1)
        
        # c) Adaptive Patch Dropout
        if self.quality_cfg.get("use_adaptive_patch_dropout", False):
            p_base = self.quality_cfg.get("patch_dropout_base_prob", 0.1)
            p_mult = self.quality_cfg.get("patch_dropout_poor_multiplier", 2.0)
            
            for i in range(x.shape[0]):
                p = p_base
                if q_labels_str[i] == "Poor":
                    p = p_base * p_mult
                elif q_labels_str[i] == "Fair":
                    p = p_base * 1.5
                
                if p > 0:
                    mask = (torch.rand(x.shape[1], device=x.device) > p).float().unsqueeze(-1)
                    x[i] = x[i] * mask

        # a) Quality Token Injection
        if hasattr(self, "quality_label_embed"):
            l_emb = self.quality_label_embed(q_indices)
            s_emb = self.quality_score_mlp(q_scores)
            q_token = self.quality_token_proj(torch.cat([l_emb, s_emb], dim=-1)).unsqueeze(1)
            x = torch.cat((cls_token, q_token, x), dim=1)
        else:
            x = torch.cat((cls_token, x), dim=1)
            
        # Positional Embedding
        pos_embed = self.backbone.pos_embed
        if x.shape[1] > pos_embed.shape[1]:
            # Interpolate or pad positional embeddings for extra tokens
            # We'll slice cls/quality from pos_embed if possible, but timm's pos_embed is [1, N, D]
            # If we added 1 token, we pad. If we added 2, we pad.
            extra = x.shape[1] - pos_embed.shape[1]
            # Simple padding with zero or repeat last (padding is safer for sanity)
            padding = torch.zeros((1, extra, pos_embed.shape[2]), device=x.device)
            pos_embed = torch.cat([pos_embed, padding], dim=1)
            
        x = x + pos_embed
        x = self.backbone.pos_drop(x)
        
        # Transformer Blocks
        x = self.backbone.blocks(x)
        
        # b) Attention Modulation (Global scaling based on quality)
        if hasattr(self, "modulation_mlp"):
            mod = self.modulation_mlp(q_indices).unsqueeze(1) # [B, 1, D]
            x = x * mod
            
        x = self.backbone.norm(x)
        
        # Classifier (using CLS token)
        return self.backbone.head(x[:, 0])

def build_vit_small(cfg: Dict[str, Any]) -> nn.Module:
    cfg["model"]["name"] = "vit_small"
    return QualityAwareRetinaViT(cfg) if cfg["model"].get("quality_aware", {}).get("enable", False) else RetinaViT(cfg)

def build_vit_base(cfg: Dict[str, Any]) -> nn.Module:
    cfg["model"]["name"] = "vit_base"
    return QualityAwareRetinaViT(cfg) if cfg["model"].get("quality_aware", {}).get("enable", False) else RetinaViT(cfg)

def build_vit_large(cfg: Dict[str, Any]) -> nn.Module:
    cfg["model"]["name"] = "vit_large"
    return QualityAwareRetinaViT(cfg) if cfg["model"].get("quality_aware", {}).get("enable", False) else RetinaViT(cfg)
