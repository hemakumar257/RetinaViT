# Project Board & Labels

Set up a GitHub Project board to track work across phases. Suggested columns:

- Backlog
- In Progress
- Review
- Done

Labels (suggested):

- `type:bug`
- `type:feature`
- `type:docs`
- `phase:1`, `phase:2`, `phase:3`, `phase:4`, `phase:5`, `phase:6`, `phase:7`, `phase:8`, `phase:9`, `phase:10`, `phase:11`, `phase:12`, `phase:13`, `phase:14`, `phase:15`

How to use:

- When creating an issue, attach the `type:*` label and the `phase:*` label corresponding to the roadmap phase.
- Move cards from Backlog → In Progress when work starts; create PRs and place cards in Review while awaiting merge.
- Close the card and move to Done when PR is merged and CI passes on `dev`/`main` as appropriate.

Creating the board:

1. Go to the repository on GitHub > Projects > New project.
2. Choose 'Board' template and add the columns above.
3. Add the suggested labels under Issues > Labels.
