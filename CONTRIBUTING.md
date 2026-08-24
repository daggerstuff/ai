# Contributing

## Development Setup

```bash
uv sync
pnpm install
```

## Safety Gate

Safety-critical modules must maintain 95% branch coverage.

```bash
uv run pytest training/tests/ --cov=training.shared_config --cov=training.multilingual_safety_checker --cov=training.clinical_safety_checker --cov=training.reward_score --cov-branch --cov-fail-under=95
```

## Commit Conventions

Include `Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>` for AI-assisted changes.
