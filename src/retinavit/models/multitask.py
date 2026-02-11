import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional

from .vit import RetinaViT, QualityAwareRetinaViT

class AttentionBlock(nn.Module):
    """
    Squeeze-and-Excitation style attention block for lesion-specific features.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DecoderBlock(nn.Module):
    """
    U-Net style decoder block with skip connection and optional attention.
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, use_attention: bool = False):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.attention = AttentionBlock(out_channels) if use_attention else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.attention(x)
        return x

class MultitaskRetinaViT(nn.Module):
    """
    Multitask Vision Transformer for joint DR classification and lesion segmentation.
    Uses a shared ViT encoder with a U-Net style segmentation decoder.
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        model_cfg = cfg["model"]
        multitask_cfg = model_cfg.get("multitask", {})
        
        # 1. Shared Encoder
        if model_cfg.get("quality_aware", {}).get("enable", False):
            self.encoder = QualityAwareRetinaViT(cfg)
        else:
            self.encoder = RetinaViT(cfg)
            
        self.backbone = self.encoder.backbone
        embed_dim = self.backbone.embed_dim
        
        # 2. Classification Head
        num_classes = model_cfg.get("num_classes", 5)
        self.classifier = nn.Linear(embed_dim, num_classes)
        
        # 3. Segmentation Decoder
        num_seg_classes = multitask_cfg.get("num_seg_classes", 1)
        use_lesion_attention = multitask_cfg.get("lesion_attention", True)
        
        # We need to extract multi-scale features. 
        # For ViT, we take intermediate block outputs.
        # Assuming we want 4 stages of features.
        self.inter_layers = multitask_cfg.get("inter_layers", [2, 5, 8, 11]) # for base-12
        
        # Decoder stages
        # stage 4: top (embed_dim) -> stage 3 skip -> stage 2 skip -> stage 1 skip
        # Note: ViT produces tokens. We reshape [B, N, D] -> [B, D, H/P, W/P]
        self.decoder_channels = multitask_cfg.get("decoder_channels", [256, 128, 64, 32])
        
        self.dec4 = DecoderBlock(embed_dim, embed_dim, self.decoder_channels[0], use_lesion_attention)
        self.dec3 = DecoderBlock(self.decoder_channels[0], embed_dim, self.decoder_channels[1], use_lesion_attention)
        self.dec2 = DecoderBlock(self.decoder_channels[1], embed_dim, self.decoder_channels[2], use_lesion_attention)
        self.dec1 = DecoderBlock(self.decoder_channels[2], embed_dim, self.decoder_channels[3], use_lesion_attention)
        
        # Final head (upsample to original res)
        self.seg_head = nn.Sequential(
            nn.Conv2d(self.decoder_channels[3], num_seg_classes, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) # Assuming final jump from P/2 to P
        )

    def extract_inter_features(self, x: torch.Tensor, quality_info: Optional[Dict[str, Any]] = None) -> List[torch.Tensor]:
        # Handle quality unaware case for simplicity in demo
        # A more robust version would handle QualityAware tokens
        
        # patch_embed
        x = self.backbone.patch_embed(x)
        cls_token = self.backbone.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.backbone.pos_embed
        x = self.backbone.pos_drop(x)
        
        features = []
        for i, block in enumerate(self.backbone.blocks):
            x = block(x)
            if i in self.inter_layers:
                features.append(x)
        
        return features

    def forward(self, x: torch.Tensor, quality_info: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        img_size = x.shape[-2:]
        p = self.backbone.patch_embed.patch_size[0]
        h, w = img_size[0] // p, img_size[1] // p
        
        # Encoder forward
        features = self.extract_inter_features(x, quality_info)
        
        # Classification (using CLS token from last layer)
        cls_feat = features[-1][:, 0]
        logits = self.classifier(cls_feat)
        
        # Segmentation Decoder forward
        # Reshape tokens to spatial maps: [B, N, D] -> [B, D, H, W]
        # Skip CLS token
        spatial_feats = []
        for feat in features:
            f = feat[:, 1:].transpose(1, 2).reshape(-1, feat.shape[-1], h, w)
            spatial_feats.append(f)
            
        # decoder stages
        d4 = self.dec4(spatial_feats[3], spatial_feats[2])
        d3 = self.dec3(d4, spatial_feats[1])
        d2 = self.dec2(d3, spatial_feats[0])
        # For d1, we might need a higher-res skip or just upsample
        d1 = self.dec1(d2, spatial_feats[0]) # Dummy skip for d1 structure
        
        mask = self.seg_head(d1)
        
        return {
            "logits": logits,
            "segmentation": mask
        }
