from retinavit.utils.config import load_config
from retinavit.utils.experiment import init_experiment, log_metrics
import time
import random

def main():
    print("--- Phase 2: Dummy Training Verification ---")
    
    # 1. Load config
    try:
        cfg = load_config(dataset="aptos", model="vit_base", training="standard")
    except Exception as e:
        print(f"Failed to load config: {e}")
        return

    # 2. Init experiment tracking
    # Set WANDB_MODE to offline for dummy run unless user has key
    import os
    if "WANDB_API_KEY" not in os.environ:
        os.environ["WANDB_MODE"] = "offline"
        print("WANDB_API_KEY not found, running in offline mode.")

    init_experiment(cfg)

    # 3. Simulate training loop
    epochs = 3
    print(f"Simulating {epochs} epochs of training...")
    for epoch in range(epochs):
        loss = 1.0 / (epoch + 1) + random.uniform(-0.1, 0.1)
        acc = 0.6 + 0.1 * epoch + random.uniform(-0.05, 0.05)
        
        metrics = {
            "loss": loss,
            "accuracy": acc,
            "epoch": epoch
        }
        
        log_metrics(metrics, step=epoch)
        print(f"Epoch {epoch}: loss={loss:.4f}, acc={acc:.4f}")
        time.sleep(1)

    print("Dummy training complete. Check W&B (offline) and MLflow (mlruns/) for logs.")

if __name__ == "__main__":
    main()
