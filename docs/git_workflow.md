# Git Workflow & Branching

Branching model:

- `main` — always stable, production-ready. Protect with branch protections and required checks.
- `dev` — integration branch for ongoing development and staging.
- `feature/<short-name>` — per-task branches (e.g., `feature/data-pipeline`).

Commit message convention (semantic prefixes):

- `feat:` — new feature (e.g., `feat: add quality-aware dataset class`)
- `fix:` — bug fix (e.g., `fix: correct ViT learning rate schedule`)
- `docs:` — documentation changes
- `refactor:` — code restructuring without behavior changes
- `chore:` — tooling, CI, repository maintenance

Pull request rules:

- Require at least 1 reviewer.
- All CI checks (lint, format, tests) must pass before merge.
- Use descriptive PR titles and link related issue/experiment IDs.

Data & privacy guidelines (do NOT commit patient data):

- Never commit raw or processed medical images to git.
- Use `data/` and `datasets/` locally but add them to `.gitignore` and never push.
- Store sensitive artifacts in secure storage (S3, Azure Blob) with access controls.

Example workflow:

```bash
# create feature branch
git checkout -b feature/vit-baseline
# implement changes, add tests
git add . && git commit -m "feat: add ViT baseline skeleton"
git push -u origin feature/vit-baseline
# Open PR to 'dev' for review
```
