# Proposal

## Why

`docs/examples` started as a collection of ideas and has become a liability:

- It consists of 16 snippets of 13 to 27 lines each, with almost no explanation.
- None of them runs as written: all of them import `Library    AIAgent`, which is not a library.
- Many of them overlap.
- The features that define the library are not documented at all: tools, toolsets and MCP, Verify, files and images, the model variable modules, and what gets logged.

The guides and the example suites in `examples/tests` are two unconnected worlds. The README is out of date as well: it still states v0.2.0 and promises a keyword bridge that does not exist yet.

## What Changes

- Add a documentation site built with Astro, using the same toolchain and structure as robotframework-jsonrpcremote. A `docs.yml` workflow publishes it to GitHub Pages.
- Establish the rule that **every guide is backed by a runnable suite in `examples/tests`**. Guides explain, suites prove. Guides embed or reference the suite instead of copying code, and the CI dry run from `add-offline-test-harness` covers every suite.
- **BREAKING** (docs only): replace `docs/examples` with a curated set of guides:
  - getting started and switching models per step
  - typed outputs and model settings
  - conversation and message history
  - files and images (`AIAgent.Content`)
  - Verify
  - the model and provider variable modules (`AIAgent.Models.*`)
  - what gets logged
  - one or two multi-agent patterns

  Superseded snippets are dropped: planner/executor and free-text spec go to the keyword bridge, typed verdict goes to Verify.
- Each feature change contributes its own guide to the site: `add-robot-tool-context`, `rework-tool-registry`, `add-keyword-bridge` and `add-agent-step-export`.
- Rewrite the README with the current status and version, links to the site, and a roadmap aligned with the open changes.
- Clean up the example suites. Playground suites (`simple2.robot`, `first.robot`, the root `tests/simple.robot`) either become guide suites or move to a `playground/` folder, which mypy already excludes.

## Capabilities

### New Capabilities

None. This change covers documentation and examples only; no library behavior changes.

### Modified Capabilities

None.

## Impact

- `docs/`: a new Astro project replaces `docs/examples`.
- `.github/workflows/docs.yml` (new); a Node.js toolchain for building the docs.
- `README.md`, `examples/README.md`, `examples/tests`, the root `tests/`.
- Basic guides can be written right after `add-offline-test-harness`. Guides for new features land with their feature changes.
