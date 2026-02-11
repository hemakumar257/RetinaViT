import os
import pandas as pd
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Any, Optional

from retinavit.data.preprocessing import FundusPreprocessor
from retinavit.data.augmentation import MedicalAugmentor
from retinavit.data.smartphone_simulation import SmartphoneSimulator
from retinavit.data.analysis import load_labels_aptos, load_labels_messidor2, load_labels_idrid

class RetinaDataset(Dataset):
    """
    Unified Dataset for APTOS, MESSIDOR-2, and IDRiD.
    """
    def __init__(
        self,
        cfg: Dict[str, Any],
        split: str = "train",
        apply_preprocessing: bool = True,
        apply_augmentation: bool = False,
        apply_simulation: bool = False
    ):
        self.cfg = cfg
        self.split = split
        self.apply_preprocessing = apply_preprocessing
        self.apply_augmentation = apply_augmentation
        self.apply_simulation = apply_simulation
        
        self.dataset_name = cfg["dataset"]["name"]
        self.data_dir = Path(cfg["paths"]["data_dir"]) / "raw" / self.dataset_name
        
        # Load labels
        self.df = self._load_data()
        
        # Load quality labels
        self.quality_df = self._load_quality_labels()
        if not self.quality_df.empty:
            # Merge quality labels if available
            # We assume image_path or image_id is the key
            # In preprocessing/scripts, we usually save filenames
            self.df = self.df.merge(self.quality_df, on="image_path", how="left")
            self.df["quality_label"] = self.df["quality_label"].fillna("Unknown")
        else:
            self.df["quality_label"] = "Unknown"
            
        # Initialize processing modules
        self.preprocessor = FundusPreprocessor(target_size=tuple(cfg["dataset"].get("target_size", (512, 512))))
        self.augmentor = MedicalAugmentor(cfg)
        self.simulator = SmartphoneSimulator(cfg)

    def _load_data(self) -> pd.DataFrame:
        """Loads dataset-specific metadata."""
        if self.dataset_name == "aptos2019":
            df = load_labels_aptos(self.data_dir)
            if not df.empty:
                df["image_path"] = df["id_code"].apply(lambda x: f"{x}.png")
                # Define synthetic patient ID for APTOS if not present
                df["patient_id"] = df["id_code"] 
            return df
        elif self.dataset_name == "messidor2":
            df = load_labels_messidor2(self.data_dir)
            if not df.empty:
                # Expecting 'image_id' column
                df["image_path"] = df["image_id"]
                # Messidor2 often has 'patient_id' in its full metadata
                if "patient_id" not in df.columns:
                    df["patient_id"] = df["image_id"].apply(lambda x: x.split('_')[0])
            return df
        elif self.dataset_name == "idrid":
            df = load_labels_idrid(self.data_dir)
            if not df.empty:
                df["image_path"] = df["Image name"].apply(lambda x: f"{x}.jpg")
                df["patient_id"] = df["Image name"]
            return df
        return pd.DataFrame()

    def _load_quality_labels(self) -> pd.DataFrame:
        """Loads quality labels from experiments/quality/."""
        quality_csv = self.cfg.get("quality_assessment", {}).get("quality_label_csv")
        if quality_csv and os.path.exists(quality_csv):
            return pd.read_csv(quality_csv)
        return pd.DataFrame()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.data_dir / row["image_path"]
        
        # In case the file doesn't exist in the raw dir (check other subfolders if needed)
        # For APTOS, images are in 'train_images' usually
        if not img_path.exists():
            if self.dataset_name == "aptos2019":
                img_path = self.data_dir / "train_images" / row["image_path"]
            elif self.dataset_name == "idrid":
                 img_path = self.data_dir / "B. Grading" / "1. Original Images" / "a. Training Set" / row["image_path"]

        image = cv2.imread(str(img_path))
        if image is None:
            # Fallback/Dummy
            image = np.zeros((512, 512, 3), dtype=np.uint8)

        # 1. Preprocessing
        if self.apply_preprocessing:
            image = self.preprocessor.preprocess_image(image)
            
        # 2. Augmentation
        if self.apply_augmentation:
            image = self.augmentor.augment(image)
            
        # 3. Smartphone Simulation
        if self.apply_simulation:
            image = self.simulator.simulate(image)
            
        # Convert to Tensor (Simple HWC -> CHW and float conversion)
        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        # Extract numerical quality score if present (fallback to 0.0)
        # We might have blur_score, illumination_score etc. We'll look for a generic 'quality_score'
        q_score = row.get("quality_score", 0.0)
        if pd.isna(q_score): q_score = 0.0
        
        return {
            "image": image,
            "label": int(row["label"]),
            "dataset": self.dataset_name,
            "quality_label": row["quality_label"],
            "quality_score": float(q_score),
            "patient_id": row["patient_id"],
            "meta": {
                "filename": row["image_path"],
                "split": self.split
            }
        }

class CombinedRetinaDataset(Dataset):
    """
    Wraps multiple RetinaDataset instances into a single unified dataset.
    """
    def __init__(self, datasets: List[RetinaDataset]):
        from typing import List
        self.datasets = datasets
        self.df = pd.concat([d.df.assign(dataset=d.dataset_name) for d in datasets], ignore_index=True)
        # Offset indices for __getitem__ mapping
        self.offsets = [0] + np.cumsum([len(d) for d in datasets]).tolist()[:-1]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Find which dataset this index belongs to
        ds_idx = 0
        for i, offset in enumerate(self.offsets):
            if idx >= offset:
                ds_idx = i
            else:
                break
        
        local_idx = idx - self.offsets[ds_idx]
        return self.datasets[ds_idx][local_idx]
