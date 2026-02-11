import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List

from retinavit.models.vit import QualityAwareRetinaViT, RetinaViT

class MetricHead(nn.Module):
    """
    Learnable metric space for medical similarity.
    Computes pairwise logits between query features and prototypes.
    """
    def __init__(self, embed_dim: int, metric_type: str = "euclidean"):
        super().__init__()
        self.metric_type = metric_type
        self.scale = nn.Parameter(torch.ones(1) * 10.0) if metric_type == "cosine" else None
        
        if metric_type == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, 1)
            )

    def forward(self, query_feats: torch.Tensor, proto_feats: torch.Tensor) -> torch.Tensor:
        # query_feats: [Nq, D]
        # proto_feats: [Nw, D]
        
        if self.metric_type == "euclidean":
            # (a-b)^2 = a^2 + b^2 - 2ab
            dist = torch.cdist(query_feats.unsqueeze(0), proto_feats.unsqueeze(0), p=2).squeeze(0)
            return -dist # Negative distance as logit
            
        elif self.metric_type == "cosine":
            query_norm = F.normalize(query_feats, p=2, dim=-1)
            proto_norm = F.normalize(proto_feats, p=2, dim=-1)
            sim = torch.mm(query_norm, proto_norm.t())
            return sim * self.scale
            
        elif self.metric_type == "mlp":
            # Pairwise concatenation
            nq = query_feats.size(0)
            nw = proto_feats.size(0)
            q_ext = query_feats.unsqueeze(1).expand(-1, nw, -1)
            p_ext = proto_feats.unsqueeze(0).expand(nq, -1, -1)
            combined = torch.cat([q_ext, p_ext], dim=-1) # [Nq, Nw, 2D]
            return self.mlp(combined).squeeze(-1)
            
        return torch.zeros(query_feats.size(0), proto_feats.size(0)).to(query_feats.device)

class CrossAttentionBlock(nn.Module):
    """
    Refines query representations by attending to support prototypes.
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, query: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        # query: [1, Nq, D], support: [1, Nw, D]
        attn_out, _ = self.mha(query, support, support)
        query = self.norm(query + attn_out)
        ffn_out = self.ffn(query)
        query = self.norm2(query + ffn_out)
        return query

class ProtoViT(nn.Module):
    """
    Prototypical Vision Transformer.
    Computes prototypes in ViT space and uses cross-attention for few-shot learning.
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        model_cfg = cfg["model"]
        proto_cfg = model_cfg.get("protovit", {})
        
        # 1. Backbone Initialization
        # Construct a backbone config derived from nested protovit config
        backbone_cfg = cfg.copy()
        backbone_cfg["model"] = model_cfg["backbone"]
        
        if backbone_cfg["model"].get("quality_aware", {}).get("enable", False):
            self.encoder = QualityAwareRetinaViT(backbone_cfg)
        else:
            self.encoder = RetinaViT(backbone_cfg)
            
        self.embed_dim = self.encoder.backbone.embed_dim
        self.feature_type = proto_cfg.get("feature_type", "class_token")
        
        # 2. Cross-Attention
        if proto_cfg.get("cross_attention", {}).get("enable", False):
            ca_cfg = proto_cfg["cross_attention"]
            self.ca_layers = nn.ModuleList([
                CrossAttentionBlock(self.embed_dim, ca_cfg.get("num_heads", 8), ca_cfg.get("dropout", 0.1))
                for _ in range(ca_cfg.get("num_layers", 1))
            ])
        else:
            self.ca_layers = None
            
        # 3. Metric Head
        self.metric_head = MetricHead(self.embed_dim, proto_cfg.get("metric_type", "euclidean"))

    def extract_features(self, x: torch.Tensor, quality_info: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        # Check if backbone is quality aware by type or presence of q-modules
        if isinstance(self.encoder, QualityAwareRetinaViT):
            # Forward through QualityAwareRetinaViT usually returns logits
            # We might need to tap into the feature extraction logic
            # For simplicity, we'll re-implement or call internal forward steps
            # Actually, let's assume we can get internal features
            return self.get_backbone_features(x, quality_info)
        else:
            return self.get_backbone_features(x)

    def get_backbone_features(self, x: torch.Tensor, quality_info: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        # For timm ViT:
        # patch_embed -> pos_embed -> blocks -> norm -> head
        
        # If QualityAware, use its forward logic up to pre-head
        if isinstance(self.encoder, QualityAwareRetinaViT):
            if quality_info is None:
                quality_info = {}
            # We copy the forward logic of QualityAwareRetinaViT but return features
            x = self.encoder.backbone.patch_embed(x)
            cls_token = self.encoder.backbone.cls_token.expand(x.shape[0], -1, -1)
            
            # (Adaptive Dropout logic can be here)
            
            # (Quality Token logic can be here)
            q_labels_str = quality_info.get("quality_label", ["Unknown"] * x.shape[0])
            q_indices = torch.tensor([self.encoder.quality_label_map.get(q, 3) for q in q_labels_str]).to(x.device)
            
            if hasattr(self.encoder, "quality_label_embed"):
                l_emb = self.encoder.quality_label_embed(q_indices)
                q_scores = quality_info.get("quality_score")
                if q_scores is None: q_scores = torch.zeros((x.shape[0], 1), device=x.device)
                elif isinstance(q_scores, list): q_scores = torch.tensor(q_scores, device=x.device).unsqueeze(-1).float()
                s_emb = self.encoder.quality_score_mlp(q_scores)
                q_token = self.encoder.quality_token_proj(torch.cat([l_emb, s_emb], dim=-1)).unsqueeze(1)
                x = torch.cat((cls_token, q_token, x), dim=1)
            else:
                x = torch.cat((cls_token, x), dim=1)

            # Positional Embed
            pos_embed = self.encoder.backbone.pos_embed
            if x.shape[1] > pos_embed.shape[1]:
                extra = x.shape[1] - pos_embed.shape[1]
                padding = torch.zeros((1, extra, pos_embed.shape[2]), device=x.device)
                pos_embed = torch.cat([pos_embed, padding], dim=1)
            
            x = x + pos_embed
            x = self.encoder.backbone.pos_drop(x)
            x = self.encoder.backbone.blocks(x)
            
            # Attention Modulation
            if hasattr(self.encoder, "modulation_mlp"):
                mod = self.encoder.modulation_mlp(q_indices).unsqueeze(1)
                x = x * mod
            
            x = self.encoder.backbone.norm(x)
        else:
            # Standard ViT
            x = self.encoder.backbone.patch_embed(x)
            cls_token = self.encoder.backbone.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            x = self.encoder.backbone.pos_drop(x + self.encoder.backbone.pos_embed)
            x = self.encoder.backbone.blocks(x)
            x = self.encoder.backbone.norm(x)
            
        if self.feature_type == "class_token":
            return x[:, 0]
        else:
            # Global Average Pooling of patch tokens (excluding cls and q tokens)
            # Offset = 1 (cls) + (1 if quality_token else 0)
            offset = 1 + (1 if hasattr(self.encoder, "quality_label_embed") else 0)
            return torch.mean(x[:, offset:], dim=1)

    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_labels: Optional[torch.Tensor] = None,
        support_quality: Optional[Dict[str, Any]] = None,
        query_quality: Optional[Dict[str, Any]] = None
    ) -> Dict[str, torch.Tensor]:
        # extraction
        support_feats = self.extract_features(support_images, support_quality) # [Ns, D]
        query_feats = self.extract_features(query_images, query_quality)      # [Nq, D]
        
        # Compute Prototypes
        # support_labels: [Ns] with values in [0, Nw-1]
        unique_labels = torch.unique(support_labels).sort()[0]
        prototypes = []
        for label in unique_labels:
            mask = (support_labels == label)
            p = support_feats[mask].mean(dim=0)
            prototypes.append(p)
        prototypes = torch.stack(prototypes) # [Nw, D]
        
        # Cross-Attention refinement
        if self.ca_layers:
            q = query_feats.unsqueeze(0) # [1, Nq, D]
            s = prototypes.unsqueeze(0)   # [1, Nw, D]
            for layer in self.ca_layers:
                q = layer(q, s)
            query_feats = q.squeeze(0)
            
        # Metric classification
        logits = self.metric_head(query_feats, prototypes) # [Nq, Nw]
        
        return {
            "logits": logits,
            "prototypes": prototypes,
            "query_features": query_feats,
            "support_features": support_feats
        }
