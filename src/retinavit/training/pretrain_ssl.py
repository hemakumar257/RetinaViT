import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any
import os

from ..models.vit import RetinaViT
from ..data.retina_dataset import CombinedRetinaDataset

class SimCLRLoss(nn.Module):
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        batch_size = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)
        z = F.normalize(z, p=2, dim=-1)
        
        sim = torch.mm(z, z.t()) / self.temperature
        mask = torch.eye(2 * batch_size, device=z.device).bool()
        sim = sim.masked_fill(mask, -1e9)
        
        targets = torch.arange(2 * batch_size, device=z.device)
        targets[:batch_size] += batch_size
        targets[batch_size:] -= batch_size
        
        return F.cross_entropy(sim, targets)

def pretrain_ssl(cfg: Dict[str, Any]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Model (Encoder only)
    model = RetinaViT(cfg).to(device)
    encoder = model.backbone
    # Simple projection head for SimCLR
    projection_head = nn.Sequential(
        nn.Linear(encoder.embed_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 128)
    ).to(device)
    
    # 2. Data
    dataset = CombinedRetinaDataset(cfg, split="train") # Mix of all datasets
    dataloader = DataLoader(dataset, batch_size=cfg.get("training", {}).get("batch_size", 32), shuffle=True)
    
    optimizer = optim.Adam(list(encoder.parameters()) + list(projection_head.parameters()), lr=1e-4)
    criterion = SimCLRLoss()
    
    model.train()
    for epoch in range(cfg.get("pretrain", {}).get("epochs", 10)):
        for i, batch in enumerate(dataloader):
            images = batch["images"].to(device)
            # Create two augmented views (Simplified: just random noise/flip for demo)
            view1 = images + torch.randn_like(images) * 0.1
            view2 = images.flip(-1) # Horizontal flip
            
            feat1 = projection_head(encoder.forward_features(view1)[:, 0])
            feat2 = projection_head(encoder.forward_features(view2)[:, 0])
            
            loss = criterion(feat1, feat2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % 10 == 0:
                print(f"Epoch {epoch} Step {i}: Loss {loss.item():.4f}")
                
    # Save pretrained weights
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(encoder.state_dict(), "checkpoints/ssl_vit_encoder.pt")
    print("Pretraining complete. Weights saved to checkpoints/ssl_vit_encoder.pt")

if __name__ == "__main__":
    # Example config for pretraining
    config = {
        "model": {"name": "vit_base", "image_size": [224, 224]},
        "training": {"batch_size": 16},
        "pretrain": {"epochs": 2},
        "data": {"datasets": ["aptos", "messidor2", "idrid"]}
    }
    # pretrain_ssl(config)
