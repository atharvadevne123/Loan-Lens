# Contributing to Loan-Lens

Thank you for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/atharvadevne123/Loan-Lens.git
cd Loan-Lens
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Running Tests

```bash
make test
```

## Code Style

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
make lint-fix
```

## Pull Request Guidelines

1. Branch from `main` with a descriptive name: `feat/shap-explainability` or `fix/drift-window`
2. Write or update tests for any changes
3. Ensure `make lint` and `make test` pass before opening a PR
4. Keep PRs focused — one feature or fix per PR

## Commit Convention

We use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat(scope): description` — new feature
- `fix(scope): description` — bug fix
- `test(scope): description` — test additions
- `chore(scope): description` — tooling / config
- `docs(scope): description` — documentation

## Reporting Issues

Please include:
- Python version and OS
- Steps to reproduce
- Expected vs actual behaviour
- Relevant log output
