import torch
import torch.nn as nn
from typing import Dict, Any, Optional

class TemporalRetinaModule(nn.Module):
    """
    Temporal model for longitudinal patient data.
    Accepts sequences of features [B, T, D] and produces patient-level predictions.
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        temporal_cfg = cfg.get("temporal", {})
        model_cfg = temporal_cfg.get("model", {})
        
        self.embed_dim = temporal_cfg.get("input_dim", 768)
        self.num_layers = model_cfg.get("num_layers", 1)
        self.num_heads = model_cfg.get("num_heads", 4)
        self.causal = model_cfg.get("causal", True)
        
        # Temporal Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.embed_dim * 2,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # Positional encoding for visits
        self.pos_embed = nn.Parameter(torch.zeros(1, 100, self.embed_dim)) # Max 100 visits
        
        # Progression head (change detection)
        # 0: Stable, 1: Worsening, 2: Improving
        self.progression_head = nn.Linear(self.embed_dim, 3)
        
        # Aggregated Patient Head (last visit representation)
        self.patient_clf = nn.Linear(self.embed_dim, 5) # Final DR grade prediction

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        x: [B, T, D] - Sequence of visit features
        mask: [B, T] - Padding mask
        """
        b, t, d = x.shape
        
        # Add positional embedding
        x = x + self.pos_embed[:, :t, :]
        
        # Causal mask for temporal modeling
        src_mask = None
        if self.causal:
            src_mask = torch.triu(torch.ones(t, t), diagonal=1).bool().to(x.device)
            
        # Transformer forward
        temp_feats = self.transformer(x, mask=src_mask, src_key_padding_mask=mask)
        
        # Per-visit progression logits
        progression_logits = self.progression_head(temp_feats)
        
        # Patient level (using last visit or mean pool)
        patient_representation = temp_feats[:, -1, :] # Last visit
        patient_logits = self.patient_clf(patient_representation)
        
        return {
            "progression_logits": progression_logits,
            "patient_logits": patient_logits,
            "temporal_features": temp_feats
        }

def build_temporal_module(cfg: Dict[str, Any]) -> TemporalRetinaModule:
    return TemporalRetinaModule(cfg)
