import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any

def distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor, 
                      temperature: float = 3.0, alpha: float = 0.5) -> torch.Tensor:
    """
    Computes the knowledge distillation loss (KL Divergence + Cross Entropy).
    """
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction='batchmean'
    ) * (temperature ** 2)
    
    hard_loss = F.cross_entropy(student_logits, labels)
    
    return alpha * hard_loss + (1 - alpha) * soft_loss

def train_distillation(teacher: nn.Module, student: nn.Module, dataloader: DataLoader, cfg: Dict[str, Any]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher.to(device).eval()
    student.to(device).train()
    
    distill_cfg = cfg.get("distill", {})
    epochs = distill_cfg.get("epochs", 5)
    lr = distill_cfg.get("lr", 1e-4)
    alpha = distill_cfg.get("alpha", 0.5)
    temp = distill_cfg.get("temperature", 3.0)
    
    optimizer = optim.Adam(student.parameters(), lr=lr)
    
    for epoch in range(epochs):
        total_loss = 0
        for i, batch in enumerate(dataloader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            with torch.no_grad():
                teacher_logits = teacher(images)
                if isinstance(teacher_logits, dict):
                    teacher_logits = teacher_logits["logits"]
                    
            student_logits = student(images)
            
            loss = distillation_loss(student_logits, teacher_logits, labels, temperature=temp, alpha=alpha)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if i % 10 == 0:
                print(f"Epoch {epoch} Step {i}: Loss {loss.item():.4f}")
        
        print(f"Epoch {epoch} finished. Avg Loss: {total_loss / len(dataloader):.4f}")
    
    print("Distillation complete.")
