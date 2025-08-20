# Robot Framework AI Agent

A Robot Framework library to talk to modern LLMs from your test cases and tasks.

## Why use this?

- Provider-agnostic: choose from OpenAI, Anthropic, Google Gemini, Vertex AI, Mistral, Groq, Cohere, Bedrock, and Hugging Face – all through pydantic-ai-slim.
- Structured outputs: ask the model to return strongly-typed results (e.g., a dataclass with fields). No more regex parsing.
- Message history made simple: by default, each `Chat` call continues from the previous call in the same test case.
- Per-call overrides: switch models and settings on the fly for a single step without re-importing the library.
- Multiple agents: import the library more than once under different aliases to create multi-agent scenarios.
- Tools and integrations: pass tools / builtin tools / toolsets from pydantic-ai-slim so agents can call functions and integrations you allow (including bridging to Robot keywords).

## Design goals

- Keep the surface area small: one core keyword (`Chat`), predefined and user defined modes and helpers.
- Be provider-agnostic and allow per-step model switching.
- Make typed outputs first-class to enable stable assertions in tests.
- Keep conversations local to a test (no surprise cross-test leakage).

## Project status and scope

The core library is intentionally small and focused. It currently exposes:

- `AIAgent.Chat` – the primary keyword to talk to LLMs (strings or typed outputs)
- History helpers:
  - `AIAgent.Get Complete History`
  - `AIAgent.Get New History`

The examples found under `examples/` are illustrative only; they are not project features.

## Compatibility

- Python: >= 3.10
- Robot Framework: >= 7.0

## Installation

`robotframework-aiagent` is a meta package, that installs everything you need to get started with the library, that means it installs every available provider and some other dependencies like mcp support.

To install `robotframework-aiagent`, run:

```bash
pip install robotframework-aiagent
```

or with `uv`:

```bash
uv pip install robotframework-aiagent
```

if you want to use specific providers or features, you can install the slim version with extras. This reduces the overall package size by excluding unused providers, features and dependencies.

```bash
# pip
pip install "robotframework-aiagent-slim[openai,mcp]"

# or with uv
uv pip install "robotframework-aiagent-slim[openai,mcp]"
```

You only need to enable extras for the providers you actually plan to use.

## Provider credentials

Set the appropriate environment variables for your chosen provider(s) before running Robot. Typical variables include (non-exhaustive):

- OpenAI: `OPENAI_API_KEY`
- Anthropic: `ANTHROPIC_API_KEY`
- Google Gemini: `GOOGLE_API_KEY`
- Vertex AI: use your GCP credentials (e.g., `GOOGLE_APPLICATION_CREDENTIALS`) and project setup
- Mistral: `MISTRAL_API_KEY`
- Cohere: `COHERE_API_KEY`
- Groq: `GROQ_API_KEY`
- AWS Bedrock: standard AWS credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, region, etc.)
- Hugging Face: `HUGGINGFACE_API_KEY` (or a token)

Refer to the respective provider documentation for the full and current requirements; pydantic-ai-slim follows the providers’ standard auth conventions.

## What you can build (use cases)

The focus is running everything directly inside Robot test cases and suites. Typical patterns:

Within that scope, useful scenarios include:

- Test data and fixtures

  - Generate realistic inputs (texts, personas, edge-case strings) for downstream steps.
  - Produce strict payloads (JSON-like) using `output_type` to avoid brittle parsing.

- Tool-augmented agents and MCP (when configured)

  - Use tools and (optionally) MCP-exposed services to fetch facts, call web services, generate JSON, or interact with systems.
  - Pair with Browser/Playwright/Selenium tools to drive UI steps directly from natural language; write the intent and execute. This can reduce upfront locator hunting if your UI library supports resilient selectors.
  - Bridge to Robot keywords via tools so agents can propose and execute actions under your allow-list.
  - Keep the final result typed for assertions in the same test.

- Information extraction and classification

  - Pull entities, tables, statuses, PII flags, or categories from unstructured logs and responses.
  - Use typed outputs for stable downstream assertions.

- Acceptance-check oracles and heuristics

  - Evaluate responses against explicit criteria; return a boolean verdict plus rationale.
  - Summarize UI/API outputs into concise, assertable statements.

- Log triage and defect reporting

  - Condense noisy logs into concise summaries; propose likely root causes or next debug steps.
  - Generate human-readable failure notes as test artifacts.

- Document QA and compliance

  - Ask targeted questions of attached docs; detect PII/compliance violations and return structured findings.

- Multimodal analysis (model-dependent)

  - Attach images/screenshots for visual QA (e.g., detect missing labels, contrast issues, text-in-image).
  - Send audio/video snippets for transcription/summary; attach documents (PDF/Doc) for summary or extraction.

- Decision routing and model selection

  - Let an agent route a case to the right sub-flow or decide which model/provider to use next.
  - Switch models locally per step without changing suite-wide defaults.

- Multi-agent workflows and simulations

  - Red-team/blue-team, user/assistant, reviewer/author; each agent can have distinct models and instructions.
  - Useful for adversarial prompts, policy reviews, or role-play testing.
  - Distribute steps across specialized agents and keep per-agent histories isolated within a test.

- Mode-specific keywords
  - Ask: question-answer turns optimized for retrieval/concise replies.
  - Edit: prompt-driven text transforms with diff/patch style outputs.
  - Agent: multi-step tool use and keyword calling.
  - Verify: structured checks and validations with boolean verdicts + rationale.
  - Custom modes: define your own keyword/mode with tailored defaults and outputs.

## Roadmap / Upcoming features

These are planned or potential additions based on the current design and dependencies:

- Tool-enabled agents (e.g., optional DuckDuckGo search)
- MCP integration
- Mode-specific keywords and defaults
  - Chat (exists today), Ask, Edit, Agent, Verify, plus custom modes
  - Edit: text transforms (summarize/translate/rewrite) and patch-like outputs
  - Agent: multi-step orchestration and tool/keyword execution
  - Verify: validations with typed verdicts
  - Extensibility: define custom modes/keywords with your own defaults and outputs
- Streaming partial responses surfaced in Robot logs during a `Chat` call
- Configurable history storage (per test/suite) and easy transcript export/attachment to Robot reports
- Global defaults via Robot variables and/or suite-level settings
- Native image and multimodal inputs where supported by the provider
- Built-in convenience schemas for common structured outputs
- Enhanced error reporting and retry policies surfaced as settings
- Optional budget manager keywords around `usage_limits` and `usage`

## Contributing

PRs and issues are welcome. Suggested setup:

```bash
# Create venv and install in editable mode with dev deps
uv pip install -e ".[dev]"

# Lint and type-check
ruff check . && ruff format --check .
mypy .

# Run example tests
robot -d results examples/tests
```

## License and links

- License: Apache-2.0
- Source: https://github.com/d-biehl/robotframework-aiagent
- Changelog: https://github.com/d-biehl/robotframework-aiagent/releases

If you have suggestions or want to contribute, issues and PRs are welcome.
