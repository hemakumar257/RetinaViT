import torch
from typing import List, Dict, Any, Tuple
from torch.utils.data import Dataset
from retinavit.data.retina_dataset import RetinaDataset

class PatientSequenceDataset(Dataset):
    """
    Groups images by patient_id to provide sequences for temporal/longitudinal analysis.
    """
    def __init__(self, retina_dataset: RetinaDataset, max_seq_len: int = 5):
        self.retina_dataset = retina_dataset
        self.max_seq_len = max_seq_len
        
        # Group indices by patient_id
        self.patient_groups = {}
        for i in range(len(self.retina_dataset.df)):
            pid = self.retina_dataset.df.iloc[i]["patient_id"]
            if pid not in self.patient_groups:
                self.patient_groups[pid] = []
            self.patient_groups[pid].append(i)
            
        self.patient_ids = list(self.patient_groups.keys())

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx) -> Dict[str, Any]:
        pid = self.patient_ids[idx]
        indices = self.patient_groups[pid]
        
        # If sequence is too long, take a sub-sequence or pad/truncate
        # For now, let's just take the first max_seq_len
        indices = indices[:self.max_seq_len]
        
        images = []
        labels = []
        q_labels = []
        
        for i in indices:
            sample = self.retina_dataset[i]
            images.append(sample["image"])
            labels.append(sample["label"])
            q_labels.append(sample["quality_label"])
            
        # Stack images into (SeqLen, C, H, W)
        images = torch.stack(images)
        labels = torch.tensor(labels)
        
        return {
            "images": images,
            "labels": labels,
            "patient_id": pid,
            "dataset": self.retina_dataset.dataset_name,
            "quality_labels": q_labels,
            "seq_len": len(indices)
        }

def get_patient_sequence(dataset: RetinaDataset, patient_id: Any) -> Dict[str, Any]:
    """Helper to retrieve all images for a specific patient."""
    indices = dataset.df.index[dataset.df['patient_id'] == patient_id].tolist()
    
    images = []
    labels = []
    for i in indices:
        sample = dataset[i]
        images.append(sample["image"])
        labels.append(sample["label"])
        
    return {
        "patient_id": patient_id,
        "images": torch.stack(images) if images else torch.empty(0),
        "labels": torch.tensor(labels) if labels else torch.empty(0),
        "seq_len": len(indices)
    }
