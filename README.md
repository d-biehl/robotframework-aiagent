# Robot Framework AI Agent

A lightweight Robot Framework library to talk to modern LLMs (via pydantic-ai-slim) from your test cases and tasks. It focuses on a clean, minimal surface: one library, one powerful keyword, and sensible defaults you can override per call.

This repository consists of a small “meta” package (`robotframework-aiagent`) that depends on the core library (`robotframework-aiagent-slim`). The core is where the Robot keywords live.

## Table of contents

- Why use this?
- Project status and scope
- Compatibility
- Installation
- Provider credentials
- Quickstart
- What you can build (use cases)
- Structured outputs (strongly typed)
- Message history and continuity
- Per-call overrides and advanced options
- Library reference (current surface)
- Running the examples (optional)
- Development
- Roadmap / Upcoming features
- FAQ / Troubleshooting
- Contributing
- License and links

## Why use this?

- Provider-agnostic: choose from OpenAI, Anthropic, Google Gemini, Vertex AI, Mistral, Groq, Cohere, Bedrock, and Hugging Face – all through pydantic-ai-slim.
- Structured outputs: ask the model to return strongly-typed results (e.g., a dataclass with fields). No more regex parsing.
- Message history made simple: by default, each `Chat` call continues from the previous call in the same test case.
- Per-call overrides: switch models and settings on the fly for a single step without re-importing the library.
- Multiple agents: import the library more than once under different aliases to create multi-agent scenarios.

## Design goals

- Keep the surface area small: one core keyword (`Chat`) and a small helper.
- Be provider-agnostic and allow per-step model switching.
- Make typed outputs first-class to enable stable assertions in tests.
- Keep conversations local to a test (no surprise cross-test leakage).
- Play nicely with standard Robot patterns (variables, aliases, logs).

## Project status and scope

The core library is intentionally small and focused. It currently exposes a single keyword `AIAgent.Chat` and one helper `AIAgent.Get Message History`. The examples found under `examples/` are illustrative only; they are not project features.

## Compatibility

- Python: >= 3.10
- Robot Framework: >= 7.0

## Installation

Install the meta package and choose the providers you want via extras:

```bash
# pip
pip install "robotframework-aiagent[openai,anthropic,google,vertexai,groq,mistral,cohere,bedrock,huggingface]"

# or with uv
uv pip install "robotframework-aiagent[openai,anthropic,google,vertexai,groq,mistral,cohere,bedrock,huggingface]"
```

If you prefer to depend on the core library directly:

```bash
pip install robotframework-aiagent-slim
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

## Quickstart

Minimal example that chats with a default model and then overrides the model for a single step.

```robot
*** Settings ***
Library    AIAgent    gpt-5-chat-latest

*** Test Cases ***
Say Hello
	AIAgent.Chat    Hello, I am a Robot Framework test.
	AIAgent.Chat    What can you do?    model=google-gla:gemini-2.5-flash-lite
```

Multiple agents with aliases (multi-agent pattern):

```robot
*** Settings ***
Library    AIAgent    gpt-5-chat-latest
Library    AIAgent    claude-sonnet-4-0    AS    SecondAgent

*** Test Cases ***
Ping Pong
	${q}    AIAgent.Chat      Ask me one question.
	${a}    SecondAgent.Chat  ${q}
	AIAgent.Chat               ${a}
```

## What you can build (use cases)

The focus is running everything directly inside Robot test cases and suites. Typical patterns:

Within that scope, useful scenarios include:

- Test data and fixtures

  - Generate realistic inputs (texts, personas, edge-case strings) for downstream steps.
  - Produce strict payloads (JSON-like) using `output_type` to avoid brittle parsing.

- Tool-augmented agents and MCP (when configured)

  - Use tools and MCP-exposed services to fetch facts, call web services, generate JSON, or interact with systems.
  - Pair with Browser/Playwright/Selenium tools to drive UI steps directly from natural language; write the intent and execute. This can reduce upfront locator hunting if your UI library supports resilient selectors.
  - Let agents call Robot keywords directly via a builtin tool.
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

### Mini example: structured classification

Ask an agent to return a typed verdict with reasoning:

```robot
*** Settings ***
Library    AIAgent
...        output_type=${{ dataclasses.make_dataclass('Verdict', [('ok', bool), ('reason', str)]) }}
...        AS    QAJudge

*** Test Cases ***
Assess Answer
	${v}    QAJudge.Chat
	...    Evaluate whether the following text meets the acceptance criteria:
	...    - Must mention an order number
	...    - Must not contain personally identifiable information
	...    Text: "Your order #A123 has shipped."
	Log    OK=${v.ok}  REASON=${v.reason}
```

Notes

- Tool/MCP support requires appropriate setup and provider/tool availability. Consult provider docs and installed extras.
- Multimodal support is model-dependent; pass the appropriate content types and keep payloads concise.

### Multi-agent workflows (specialized agents)

Define multiple agents with clear roles and distribute steps across them. Each agent keeps its own message history within a test case, so conversations don’t mix. Typical patterns:

- Extractor → Judge → Reporter: one agent extracts structured data, another validates/complies, a third produces a human summary.
- Planner → Executor: a planning agent proposes next actions; an execution agent performs them (pair Executor with other Robot keywords).
- Author ↔ Reviewer: content generation with immediate review for tone/style/policy alignment.
- Red ↔ Blue: adversarial prompts vs. defender analysis to probe robustness.
- Ensemble and arbitration: query multiple providers/agents and have an arbiter agent pick the best response.
- Fallback routing: try a cheap/fast model first; escalate to a stronger model only if confidence is low.

Compact example:

```robot
*** Settings ***
Library    AIAgent
...        output_type=${{ dataclasses.make_dataclass('Data', [('order_id', str)]) }}
...        AS    Extractor
Library    AIAgent
...        output_type=${{ dataclasses.make_dataclass('Verdict', [('ok', bool), ('reason', str)]) }}
...        AS    Judge
Library    AIAgent    gpt-5-nano    AS    Reporter

*** Test Cases ***
Chain Of Responsibility
	${data}     Extractor.Chat    Extract the order id from: "Order #A123 has shipped to Berlin."
	${verdict}  Judge.Chat        Check if the text reveals PII. Return ok + reason.
	...                          The order id is ${data.order_id}.
	${summary}  Reporter.Chat
	...                          Summarize the outcome for a human reader based on:
	...                          - Order ID: ${data.order_id}
	...                          - Verdict OK: ${verdict.ok}
	...                          - Reason: ${verdict.reason}
	Log         ${summary}
```

More patterns with compact examples:

1. Planner → Executor loop

```robot
*** Settings ***
Library    AIAgent    AS    Planner
Library    AIAgent    AS    Executor

*** Test Cases ***
Plan And Do (Loop)
	${Plan}    Set Variable    ${{ dataclasses.make_dataclass('Plan', [('action', str), ('arg', str)]) }}
	FOR    ${i}    IN RANGE    3
		${step}    Planner.Chat
		...    Propose the next action as a tuple (action, arg). Keep it simple.
		...    Examples: Click(#submit), Assert("Logged in").
		...    Return only the action and arg.
		...    output_type=${Plan}

		${result}    Executor.Chat
		...    Execute: ${step.action} ${step.arg}
		...    Respond with a brief status line.

		# Optionally break on success keyword in ${result}
	END
```

2. Author ↔ Reviewer with style/policy

```robot
*** Settings ***
Library    AIAgent    AS    Author
Library    AIAgent
...        output_type=${{ dataclasses.make_dataclass('Review', [('ok', bool), ('edits', str), ('reason', str)]) }}
...        AS    Reviewer

*** Test Cases ***
Draft And Review
	${draft}    Author.Chat    Write a 3-sentence product update note.
	${r}        Reviewer.Chat  Review for clarity and tone. If not ok, propose edits.    ${draft}
	IF    not ${r.ok}
		${draft}    Author.Chat    Apply these edits: ${r.edits}
	END
```

3. Ensemble and arbitration

```robot
*** Settings ***
Library    AIAgent    gpt-5-nano          AS    A1
Library    AIAgent    claude-sonnet-4-0   AS    A2
Library    AIAgent    google-flash-2-4
...        output_type=${{ dataclasses.make_dataclass('Choice', [('best', int), ('reason', str)]) }}
...        AS    Arbiter

*** Test Cases ***
Pick Best Answer
	${q}     Set Variable    Explain the difference between mocks and stubs.
	${r1}    A1.Chat    ${q}
	${r2}    A2.Chat    ${q}
	${c}     Arbiter.Chat
	...    Choose best=1 for A1 or best=2 for A2.
	...    Consider correctness and clarity. Return best and reason.
	Log     Best=${c.best}  Reason=${c.reason}
```

4. Fallback routing by confidence

```robot
*** Settings ***
Library    AIAgent    gpt-5-nano    AS    Fast
Library    AIAgent                 AS    Strong

*** Test Cases ***
Cheap Then Strong
	${Verdict}    Set Variable    ${{ dataclasses.make_dataclass('Verdict', [('confident', bool), ('answer', str)]) }}
	${v}    Fast.Chat    Answer briefly. Return confident + answer.    output_type=${Verdict}
	IF    not ${v.confident}
		${answer}    Strong.Chat    Provide a precise, verified answer.
	ELSE
		${answer}    Set Variable    ${v.answer}
	END
	Log    ${answer}
```

Tips

- Each agent’s history is independent and reset at end of test. Use `message_history=None` to force a fresh turn.
- Mix providers per agent (e.g., Extractor on a fast local model, Judge on a higher-accuracy cloud model).

### From free-text spec to executable steps (guarded autonomy)

You can accept a natural-language spec and let an agent propose concrete Robot steps to execute—while enforcing safety via allow-lists and typed outputs. Pattern:

1. Constrain what’s allowed (list of keywords/tools, argument formats, timeouts).
2. Have the agent return one structured action at a time.
3. Execute the action with `Run Keyword` and feed the outcome back to the agent.

Example loop:

```robot
*** Settings ***
Library    AIAgent    AS    Planner

*** Test Cases ***
Execute Free-Text Spec
	${Spec}    Set Variable    As a user, open the demo app, log in as "alice", then verify greeting "Welcome, Alice".
	${Step}    Set Variable    ${{ dataclasses.make_dataclass('Step', [('keyword', str), ('args', list[str]), ('done', bool)]) }}

	WHILE    ${True}
		${s}    Planner.Chat
		...    Read the spec and propose the next Robot step as Step(keyword, args, done).
		...    Allowed keywords: Browser.Open, Browser.Fill Text, Browser.Click, Should Contain.
		...    Use only allowed keywords and valid args. No prose.
		...    output_type=${Step}

		IF    ${s.done}    BREAK
		Run Keyword    ${s.keyword}    @{s.args}

		# Optionally provide a short status back for planning continuity
		Planner.Chat    Executed: ${s.keyword}    message_history=${None}
	END
```

Notes

- Keep an allow-list narrow and validated; consider dry-runs in CI.
- For direct browser/UI control, combine with Browser/Playwright/other libraries in the allow-list.
- For web services/JSON, allow keywords that call HTTP or your internal client wrappers.

### Practical tips

- Reliability: keep temperature low, write clear instructions, and use `output_type` for structured outputs.
- Cost/runtime: set `usage_limits`, keep prompts concise, and pick efficient models for simple tasks.
- Context: control `message_history` explicitly in loops and multi-agent exchanges to avoid unintended carryover.

### Quick examples

- Per-step model switch:

```robot
AIAgent.Chat    Summarize briefly.    model=claude-sonnet-4-0
AIAgent.Chat    Now translate to German.    model=google-gla:gemini-2.5-flash-lite
```

- Multimodal attachment (model-dependent):

```robot
# Pseudocode-style; exact attachment syntax depends on model/provider support
${img}      Get File    screenshots/login.png
${summary}  AIAgent.Chat    Analyze this screenshot for missing labels and contrast issues.
...                          attachments=${img}
Log         ${summary}
```

## Configuration by example

You can set most defaults at library import and override per call:

```robot
*** Settings ***
Library    AIAgent
...        model=gpt-5-chat-latest
...        instructions=You are a helpful QA assistant.
...        system_prompt=Answer concisely.
...        output_type=${{ str }}
...        retries=1
...        output_retries=1
```

Notes

- `model_settings` can pass provider-specific parameters (e.g., temperature). Use per-call when tuning a single step.
- To use a local/OpenAI-compatible backend, set `model` to the identifier your backend expects and ensure env vars/endpoint are configured.

## Structured outputs (strongly typed)

You can request the model to return a typed structure. The library forwards an `output_type` to pydantic-ai-slim and returns the parsed object. A convenient approach in Robot is to create a dataclass via inline evaluation and set it as the `output_type` either at library import time or per call.

At library import time:

```robot
*** Settings ***
Library    AIAgent    gpt-5-nano
...        output_type=${{ dataclasses.make_dataclass('Result', [('should_break', bool), ('reason', str)]) }}
...        AS    SemanticAgent

*** Test Cases ***
Decide End Of Conversation
	${result}    SemanticAgent.Chat    Please return whether this conversation should end.
	Log    ${result.should_break}: ${result.reason}
```

Or override per call:

```robot
${schema}    Set Variable    ${{ dataclasses.make_dataclass('Answer', [('value', int)]) }}
${answer}    AIAgent.Chat    Give me a number between 1 and 10.    output_type=${schema}
Log    ${answer.value}
```

Note: Robot can access dataclass attributes with the dot notation as shown above.

## Message history and continuity

By default, `AIAgent.Chat` continues from the previous call within the same test case. This behavior is controlled by the `message_history` argument, which defaults to `LAST_RUN`.

- To start fresh for a call, set `message_history=None`.
- To provide your own history, pass a list of model messages (advanced; most users won’t need this).
- The library resets its internal “last run result” at the end of each test case, so conversations do not leak across tests.

You can also retrieve the last run’s messages explicitly:

```robot
${history}    AIAgent.Get Message History
Log Many      ${history}
```

## Per-call overrides and advanced options

The `Chat` keyword accepts a set of optional named arguments to fine-tune behavior for individual steps:

- `model`: override the model just for this call. Accepts provider-qualified strings (e.g., `google-gla:gemini-2.5-flash-lite`) or known model names.
- `model_settings`: provider-specific parameters such as temperature or token limits.
- `output_type`: override the output schema for this call.
- `usage_limits` / `usage`: token/request budgeting objects from pydantic-ai-slim if you want to cap usage.
- `infer_name`: toggle model-specific name inference.

Examples:

```robot
# Tweak model settings for a single step
${settings}    Set Variable    ${{ {'temperature': 0.2, 'max_output_tokens': 256} }}
AIAgent.Chat    Summarize this text in one sentence.    model_settings=${settings}

# Switch the model only for this step
AIAgent.Chat    Answer concisely.    model=claude-sonnet-4-0
```

## Library reference (current surface)

Library class: `AIAgent`

Constructor parameters (when importing the library):

- `model`: default model for the agent (string or known name)
- `output_type`: default output schema (defaults to `str`)
- `instructions`: optional additional instructions
- `system_prompt`: optional system prompt (string or list of strings)
- `name`: agent name
- `model_settings`: default provider settings (dict-like)
- `retries`: model call retries
- `output_retries`: structured output parsing retries

Keywords:

- `AIAgent.Chat` – sends user content to the model and returns the output (string or typed object). Signature highlights: `output_type`, `message_history`, `model`, `model_settings`, `usage_limits`, `usage`, `infer_name`.
- `AIAgent.Get Message History` – returns the messages from the last completed `Chat` run (or `None`).

## Running the examples (optional)

The `examples/` directory contains Robot test suites demonstrating usage patterns (e.g., multi-agent, structured outputs). These are examples only and not project features.

```bash
# Run example tests and store results under ./results
robot -d results examples/tests
```

## Development

This repository uses Hatch + uv-dynamic-versioning. The root package (`robotframework-aiagent`) pins and forwards the version to the core (`robotframework-aiagent-slim`). Linting and typing use Ruff, MyPy, and Pyright.

Useful tips:

- Make sure you’re on Python 3.10+.
- Install dev tools: `uv pip install -e ".[dev]"` (or use your preferred manager).
- Lint/typecheck: Ruff/Pyright/MyPy configs are included.

## Roadmap / Upcoming features

These are planned or potential additions based on the current design and dependencies:

- Tool-enabled agents (e.g., optional DuckDuckGo search)
- MCP integration
- Mode-specific keywords and defaults
  - Ask: optimized Q&A
  - Edit: text transforms (summarize/translate/rewrite) and patch-like outputs
  - Agent: multi-step orchestration and tool/keyword execution
  - Verify: validations with typed verdicts
  - Extensibility: define custom modes/keywords with your own defaults and outputs
- Streaming updates surfaced in Robot logs during a `Chat` call
- Persistent memory across test cases or suites (configurable)
- Easy transcript export and attachment to Robot reports
- Global defaults via Robot variables and/or suite-level settings
- Native image and multimodal inputs where supported by the provider
- Built-in convenience schemas for common structured outputs
- Better error reporting and retry policies surfaced as settings
- Optional budget manager keywords around `usage_limits` and `usage`

## FAQ / Troubleshooting

- I get 401/permission errors.
  - Check the correct API key env var for your provider; verify the project/region for Vertex/AWS.
- The model output doesn’t match my schema.
  - Tighten instructions, reduce temperature, and consider increasing `output_retries` at import.
- Conversations bleed across tests.
  - By design, agent state resets at end of test. If you see unexpected carryover within a test, set `message_history=None` on the call that should start fresh.
- How do I switch between providers or local models?
  - Set a per-call `model` string (e.g., `model=claude-sonnet-4-0`). For local OpenAI-compatible servers, point to the correct model name and ensure the client env vars/URL are set.
- Can I attach images/audio/documents?
  - This is model-dependent and may require additional wiring; treat the example as guidance and consult your provider’s docs.

## Contributing

PRs and issues are welcome. Suggested setup:

```bash
# Create venv and install in editable mode with dev deps
uv pip install -e ".[dev]"

# Lint and type-check
ruff check . && ruff format --check .
pyright
mypy

# Run example tests
robot -d results examples/tests
```

## Known limitations

- The public surface is intentionally small: primarily `Chat` and a helper to retrieve the last message history.
- Message history is reset at the end of each test case (by design to avoid leakage across tests).
- Provider APIs and model names change quickly; prefer passing explicit model strings you control.

## License and links

- License: Apache-2.0
- Source: https://github.com/d-biehl/robotframework-aiagent
- Changelog: https://github.com/d-biehl/robotframework-aiagent/releases

If you have suggestions or want to contribute, issues and PRs are welcome.
