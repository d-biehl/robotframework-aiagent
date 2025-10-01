# Repository Guidelines

## Project Structure & Module Organization
The workspace relies on uv for a multi-package layout. Core library code lives in `packages/slim/src/AIAgent`, where the `Agent/` subpackage hosts runtime keywords. Robot Framework suites and their resources are under `examples/tests`, while supporting Python utilities sit in `examples/src`. Long-form guides are kept in `docs/examples`, prompt assets in `prompts/`, and generated Robot outputs should remain in `results/` for easy diffing.

## Build, Test, and Development Commands
Bootstrap a local environment with `uv pip install -e ".[dev]"`. Run static checks using `uv run ruff check .` and enforce formatting with `uv run ruff format --check .`. Type analysis is required via `uv run mypy .` (or `uv run pyright` if you want the alternative report). Execute the acceptance suites with `uv run robot -d results examples/tests`, narrowing to a single `.robot` file when debugging.

## Coding Style & Naming Conventions
Code targets Python 3.10, four-space indentation, and Ruff’s 120-character limit. Ruff also enforces single-quoted strings and Google-style docstrings where applicable, so prefer docstrings over comments for keyword documentation. Treat `AIAgent` as the first-party import root, keep module filenames lowercase with underscores, and reserve PascalCase for classes. Reach for `typing_extensions.TypedDict` when exposing structured outputs back to Robot.

## Testing Guidelines
Robot suites in `examples/tests` double as regression coverage; name files after the scenario (e.g., `ensemble and arbitration.robot`) and place shared keywords in `examples/tests/resources`. Always run `uv run robot -d results examples/tests` before opening a PR and inspect the refreshed `results/output.xml` and logs. Supplement with focused unit tests near the Python module you touched and confirm strict type coverage with mypy.

## Commit & Pull Request Guidelines
Follow the conventional commit style already present (`feat:`, `fix:`, `docs:`, etc.) and write concise, imperative summaries. Every pull request should include a brief what/why, links to related issues, and a checklist of verification commands you executed. Attach relevant artifacts—trimmed logs, Robot reports, or screenshots—when behavior or tooling changes.

## Agent Configuration Tips
Shared defaults live in `robot.toml`; extend modes or tools there when you add capabilities. Keep secrets out of source control and rely on provider-specific environment variables noted in `README.md`. If you contribute new prompt templates to `prompts/`, describe their intent and expected outputs in the accompanying PR discussion.
