# Proposal

## Why

Nothing in the repository is verified automatically: the only workflow publishes releases, there are no unit tests, and the Robot suites need real API keys. Regressions go unnoticed. Every guide in `docs/examples` imports a library that does not exist, and log messages written by tools silently vanish. The upcoming refactors (synchronous `Chat`, tool registry, keyword bridge) need a safety net that runs without network access or API keys.

## What Changes

- Add Python unit tests that use pydantic-ai's `TestModel` and `FunctionModel`, so they need no network access and no API keys. They cover:
  - `AgentRunner`: runs, message history, timeouts and propagation of failed runs
  - log rendering (`log_format`)
  - plugin discovery (`PluginManager`)
  - the `Verify` library
- Add an offline set of Robot suites that import `AIAgent.Agent` with a test model object instead of a provider model. This exercises the keywords end to end inside Robot Framework.
- Run `robot --dryrun` over all example suites. This catches unknown libraries, keywords and argument errors without calling any model.
- Add a CI workflow (`ci.yml`) modeled on robotframework-jsonrpcremote:
  - checks: `ruff check`, `ruff format --check`, `mypy`, unit tests, offline suites and the dry run
  - matrix: ubuntu, macos and windows × Python 3.12, 3.13, 3.14, 3.14t and 3.15
- **BREAKING**: raise `requires-python` to `>=3.12` for all workspace packages, and align ruff `target-version` and the mypy/pyright `python_version` with it.
- Add `pytest` as a dev dependency. Document the test commands in `AGENTS.md` and in the README's contributing section.

## Capabilities

### New Capabilities

None. This change adds test and CI infrastructure only; no library behavior changes.

### Modified Capabilities

None.

## Impact

- `pyproject.toml` (root, `packages/slim`, `packages/verify`, `examples`): `requires-python`, tool targets, dev dependencies.
- `.github/workflows/ci.yml` (new).
- New unit tests and offline suites. Their location is decided in design. The existing `tests/simple.robot` is a key-requiring playground and does not belong in the offline set.
- The free-threaded 3.14t job depends on pydantic-ai and the provider SDKs running without the GIL. Design decides whether that job is allowed to fail at first.
