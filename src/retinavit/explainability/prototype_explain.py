import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt

def compute_prototype_similarity_maps(
    model: torch.nn.Module, 
    support_images: torch.Tensor,
    query_images: torch.Tensor,
    support_labels: torch.Tensor,
    query_labels: torch.Tensor,
    device: torch.device
) -> Dict[str, Any]:
    """
    Computes spatial similarity maps between query features and prototypes.
    """
    model.eval()
    with torch.no_grad():
        # Get prototypes and query features
        # We need to extract spatial query features [B, D, H, W]
        outputs = model(
            support_images.to(device), support_labels.to(device),
            query_images.to(device), query_labels.to(device),
            return_spatial=True # We assume the model supports this flag or we extract manually
        )
        
        prototypes = outputs["prototypes"] # [C, D]
        query_spatial_feat = outputs["query_spatial_features"] # [B, D, H, W]
        
        # Compute cosine similarity per pixel
        # Normalize prototypes
        prototypes = F.normalize(prototypes, p=2, dim=-1) # [C, D]
        
        # Normalize spatial features
        b, d, h, w = query_spatial_feat.shape
        feat_flat = query_spatial_feat.view(b, d, -1).transpose(1, 2) # [B, HW, D]
        feat_flat = F.normalize(feat_flat, p=2, dim=-1)
        
        # Similarity: [B, HW, C]
        sim_maps = torch.bmm(feat_flat, prototypes.unsqueeze(0).repeat(b, 1, 1).transpose(1, 2))
        sim_maps = sim_maps.view(b, h, w, -1).permute(0, 3, 1, 2) # [B, C, H, W]
        
        return {
            "similarity_maps": sim_maps.cpu().numpy(),
            "query_preds": outputs["logits"].argmax(dim=1).cpu().numpy()
        }

def get_similar_support_examples(
    model: torch.nn.Module,
    query_image: torch.Tensor,
    support_images: torch.Tensor,
    support_labels: torch.Tensor,
    device: torch.device,
    top_k: int = 3
) -> List[Tuple[int, float, int]]:
    """
    Finds the top-k most similar support images for a given query image.
    Returns list of (index, similarity_score, label)
    """
    model.eval()
    with torch.no_grad():
        # Extract global features
        q_feat = model.extract_features(query_image.to(device)) # [1, D]
        s_feats = model.extract_features(support_images.to(device)) # [S, D]
        
        q_feat = F.normalize(q_feat, p=2, dim=-1)
        s_feats = F.normalize(s_feats, p=2, dim=-1)
        
        cos_sim = torch.mm(q_feat, s_feats.t()).squeeze(0) # [S]
        
        top_scores, top_idxs = torch.topk(cos_sim, k=min(top_k, len(cos_sim)))
        
        results = []
        for score, idx in zip(top_scores, top_idxs):
            results.append((int(idx), float(score), int(support_labels[idx])))
            
        return results

def visualize_similarity_grid(query_img, similar_examples, support_imgs, labels_map):
    """
    Helper to visualize a query image alongside its most similar support cases.
    """
    n = len(similar_examples) + 1
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    
    # Query
    axes[0].imshow(query_img)
    axes[0].set_title("Query Image")
    axes[0].axis("off")
    
    # Support
    for i, (idx, score, label) in enumerate(similar_examples):
        axes[i+1].imshow(support_imgs[idx])
        lbl_name = labels_map.get(label, str(label))
        axes[i+1].set_title(f"Support {idx}\nScore: {score:.2f}\nLabel: {lbl_name}")
        axes[i+1].axis("off")
        
    return fig
