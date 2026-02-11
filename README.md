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
python -m venv .venv
source .venv/bin/activate     # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Run tests:

```bash
./scripts/run_tests.sh
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
