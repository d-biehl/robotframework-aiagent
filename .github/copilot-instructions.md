## Copilot instructions for robotframework-aiagent

### Purpose
Help AI coding agents work productively in this repo by capturing how it’s structured, how to run it, and the patterns the code relies on.

### Big picture
- This repo has a tiny “meta” package (`robotframework-aiagent`) and the actual core library in `packages/slim` (`robotframework-aiagent-slim`). Code lives under `packages/slim/src/AIAgent`.
- The library exposes a Robot Framework library class `AIAgent` with one primary keyword: `Chat`. It wraps pydantic-ai-slim to talk to many model providers and supports strongly typed outputs.
- Message history is per-test by design. The library resets its last run state at end of each test case (listener=SELF).

### Key files and where to look
- Core library: `packages/slim/src/AIAgent/__init__.py` (keywords, message history handling, toolset support).
- Usage examples: `examples/tests/*.robot`, example tools in `examples/src/AIAgentExamples/WeatherTools.py`.
- Root config and tooling: `pyproject.toml` (workspace, dev tooling, ruff/pyright/mypy), `packages/slim/pyproject.toml` (runtime deps and extras).

### How AIAgent works (mental model)
- `@library(..., listener='SELF')` keeps state inside the suite and clears `_last_run_result` on `end_test`.
- `Chat` streams a model run and finally logs and returns `result.output` (string or typed object). Default `message_history=LAST_RUN` continues the previous call within the same test.
- You can pass provider-specific settings (`model`, `model_settings`), usage budgeting, and `output_type` per call. The constructor also accepts defaults.
- Tooling integration: constructor and `Chat` accept `tools`, `builtin_tools`, and `toolsets` from pydantic-ai-slim. See `WeatherTools.py` for a `FunctionToolset` example.

### Conventions and patterns
- Multi-agent: import `AIAgent` multiple times with `AS` to isolate roles; each agent holds its own last-run history within the test.
- Typed outputs: prefer `output_type` (e.g., dataclasses) for stable assertions in Robot.
- Known model names: `KnownModelName` mirrors pydantic-ai-slim’s known models; you can also pass plain strings like `google-gla:gemini-2.5-flash-lite`.
- Logging: final model output is written via `robot.api.logger.info`; keep this behavior for visibility in logs/reports.

### Local dev workflow
- Python 3.10+; uses uv + hatch dynamic versioning; workspace members: `packages/*`, `examples`.
- Install dev deps: editable root with extras as needed; run linters and type-checkers defined in `pyproject.toml` (ruff, pyright, mypy).
- Quick smoke: run example suites and inspect `results/`.

### Typical commands (zsh)
- Install (editable dev): uv/pip against root with provider extras in the meta package.
- Lint/type: ruff check/format, pyright, mypy.
- Run examples: `robot -d results examples/tests`.

### Integration notes
- Providers and tools come from pydantic-ai-slim. Ensure API keys/environment are set (e.g., OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.) before running Robot.
- Message history helpers exist: `Get All Run Result Messages` and `Get New Run Result Messages` return raw or JSON-serialized message lists.

### When changing the library
- Preserve `listener='SELF'` and the `end_test` reset to prevent cross-test leakage.
- Keep the `converters` for toolsets so Robot can pass pydantic toolsets.
- Maintain `Chat` return semantics (return parsed output and log it) and defaults (`message_history=LAST_RUN`).

### Good starting points
- Start from `examples/tests/*.robot` to see library usage and multi-agent patterns.
- Trace the `Agent` creation in `AIAgent.agent` for default behaviors and settings.

### Questions or gaps?
If any workflow or convention above is unclear, highlight it and we’ll iterate on these instructions.
