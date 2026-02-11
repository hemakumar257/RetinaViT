# RetinaViT: Few-Shot, Quality-Aware Vision Transformer for Retinal Imaging

RetinaViT is a research-grade codebase for developing few-shot, quality-aware Vision Transformer (ViT) models for fundus/retinal imaging. The project focuses on robust clinical-grade modeling, explainability, and lightweight deployment for mobile and web platforms.

## Features
- Data pipeline for fundus images with quality-aware filtering and augmentation
- Vision Transformer and prototype-based few-shot variants (ProtoViT)
- Explainability hooks and saliency-map utilities
- Experiment logging and reproducible configs
- Model export and lightweight deployment tooling for mobile/web

## Project Structure
- `src/` — Python package and source code (models, data loaders, training loops)
- `configs/` — YAML configuration files for datasets, models and training runs
- `experiments/` — Experiment naming and logging guidance
- `docs/` — Documentation, contributing and workflow guides
- `tests/` — Unit and integration tests
- `scripts/` — Helper scripts (env setup, run tests)
- `.github/` — CI/CD and issue templates

## Getting Started
Create a Python virtual environment and install development dependencies:

```bash
# Run the setup script
bash scripts/setup_env.sh

# Or manually:
python -m venv .venv
source .venv/bin/activate     # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Run tests:

```bash
pytest tests/
```

## Datasets
The project utilizes three primary datasets for training and validation:
- **APTOS 2019**: Blindness detection classification.
- **MESSIDOR-2**: Diabetic retinopathy screening.
- **IDRiD**: Segmentation and grading.

**Note**: Raw data is not included in this repository due to size and licensing. Please refer to [docs/datasets/licenses.md](docs/datasets/licenses.md) and the individual dataset documentation in `docs/datasets/` for acquisition instructions.

## Experiment Tracking
We use Weights & Biases (W&B) and MLflow for tracking.
- **W&B**: Set your API key: `export WANDB_API_KEY=your_key`. Runs in `offline` mode by default if no key is found.
- **MLflow**: Logs are stored locally in the `mlruns/` directory by default.

To verify the integration, run the dummy training script:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python src/retinavit/training/train_dummy.py
```

## Roadmap
- Phase 1: Project Infrastructure & Git Foundation
- Phase 2: Data ingestion and preprocessing pipelines
- Phase 3: Baseline ViT implementation
- Phase 4: Few-shot prototype module (ProtoViT)
- Phase 5: Quality-aware augmentation and filtering
- Phase 6: Evaluation suite and metrics dashboard
- Phase 7: Explainability & saliency modules
- Phase 8: Model compression and mobile optimization
- Phase 9: End-to-end training & hyperparam sweeps
- Phase 10: Clinical validation and safety checks
- Phase 11: Deployment & CI for model serving
- Phase 12: Web demo and mobile app integration
- Phase 13: Regulatory & privacy compliance documentation
- Phase 14: Benchmarking and leaderboards
- Phase 15: Community adoption and data partnerships

---
Project repository: https://github.com/hemakumar257/RetinaViT
