# Experiments

Experiment naming and logging guidance:

- Use structured names: `phaseX_experiment_shortdesc` (e.g., `phase3_vit_baseline_lr1e-4`)
- Log configs (YAML) alongside run artifacts and seed values for reproducibility.
- Use centralized tracking (Weights & Biases, MLflow) configured to not store PHI.
