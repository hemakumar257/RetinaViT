#!/bash/bin

# 1. Run Self-Supervised Pretraining
python -m retinavit.training.pretrain_ssl

# 2. Run HPO for Multitask
python -m retinavit.training.hpo
