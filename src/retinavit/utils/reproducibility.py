import torch
import numpy as np
import random
import os
import sys

def seed_everything(seed: int = 42):
    """
    Ensures reproducibility by seeding all random generators.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Global seed set to: {seed}")

def freeze_environment(file_path: str = "docs/release/environment_freeze.txt"):
    """
    Saves a snapshot of the current environment state.
    """
    import subprocess
    result = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        f.write("# Environment frozen for RetinaViT Phase 15 Release\n")
        f.write(result.stdout)
    print(f"Environment freeze saved to {file_path}")

if __name__ == "__main__":
    seed_everything(42)
    freeze_environment()
