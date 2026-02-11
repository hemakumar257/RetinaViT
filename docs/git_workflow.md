# Git Workflow Conventions

## Branching Model
- **main**: Stable production-ready branch.
- **dev**: The main integration branch for all new features.
- **feature/<short-name>**: Short-lived branches for specific features or fixes (e.g., `feature/data-loader`).

## Commit Message Format
We follow the conventional commits specification:
- `feat: ...` for new features
- `fix: ...` for bug fixes
- `docs: ...` for documentation changes
- `refactor: ...` for code changes that neither fix a bug nor add a feature
- `chore: ...` for maintenance tasks

## Pull Request Rules
- At least one reviewer must approve the PR.
- All CI checks (linting and tests) must pass.
- Squash and merge is preferred for feature branches.

## Data Governance
- **NEVER** commit raw or processed medical images to the repository.
- Use `data/` and `datasets/` folders locally; these are ignored by Git.
- Use synthetic or anonymized data for testing if necessary.
