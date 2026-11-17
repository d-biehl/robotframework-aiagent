# Proposal

## Why

Robot Framework's strength is its library ecosystem: Browser, RequestsLibrary, SSHLibrary, DatabaseLibrary and a project's own `.resource` files. Today an agent can only use Python tools or MCP servers. It cannot drive the browser session the test itself opened. `docs/examples/09` works around this with a hand-written planner loop around `Run Keyword`. Letting agents call Robot keywords directly makes the whole ecosystem available as tools. It also turns documented user keywords into tools that need neither Python nor packaging. The README already promises this bridge.

## What Changes

- Add a keyword bridge toolset that an agent can enable; the exact option is decided in design. Instead of one tool per keyword, it exposes a fixed set of meta tools, so the prompt does not grow with the number of loaded libraries:
  - list the visible libraries, including resource files and the suite's own keywords
  - list the keywords of a library or resource, with name and short documentation
  - get the full documentation of a library or keyword, including its argument specification (types, defaults)
  - search keywords across all visible libraries by text
  - run a keyword:
    - positional and named arguments are given as strings, and Robot performs its usual argument conversion
    - an optional `assign` stores the return value as `${name}`, and later calls may reference it
- Argument handling only substitutes plain `${name}` references. Inline Python (`${{ }}`) and the extended variable syntax are never evaluated.
- Visibility: everything loaded in the current scope is visible, minus a deny list. The deny list combines built-in defaults with user patterns (globs over `Library.Keyword`). Users can explicitly re-allow default entries.
- The default deny list blocks:
  - **BuiltIn flow control**: Fatal Error, Pass Execution (If), Skip (If), Return From Keyword (If), Exit/Continue For Loop (If), Run Keyword And Return (If)
  - **BuiltIn keywords that bypass the deny list**: all `Run Keyword*` variants, Run Keywords, Repeat Keyword, Wait Until Keyword Succeeds
  - **BuiltIn configuration**: Import Library/Resource/Variables, Reload Library, Set Library Search Order, Set/Reset Log Level
  - **BuiltIn result manipulation**: Set/Remove Tags, Set Test Message, Set Test/Suite Documentation, Set Test/Suite/Task Metadata
  - **BuiltIn variables beyond the test**: Set Global Variable, Set Suite Variable
  - **BuiltIn variable disclosure**: Get Variables, Log Variables, Get Variable Value
  - **BuiltIn code execution**: Evaluate, Call Method, Get Library Instance. Should (Not) Be True and Set Variable If stay allowed on purpose, because they are useful assertions.
  - **OperatingSystem**: Run, Run And Return Rc (And Output), Remove Directory/File/Files, Empty Directory, and the environment variable keywords
  - **Process**: Run Process, Start Process, Send Signal To Process
  - **Dialogs**: all keywords
  - the calling agent's own `Chat`, to prevent recursion
- Keywords run in Robot's main thread through the work queue from `add-robot-tool-context`. They appear in the log nested under `Chat` with their own status, and async keywords run natively, so `Chat` of another agent works as a tool.
- Failure handling is configurable at library import and per `Chat` call:
  - default: the failure message goes back to the model as the tool result; the keyword shows FAIL in the log and the agent can react
  - alternative: the failure ends `Chat` and fails the test immediately
- Out of scope: one tool per keyword with deferred loading through pydantic-ai's `ToolSearch`. That is a possible later optimization if the round trips of the meta tools become a problem.

## Capabilities

### New Capabilities

- `keyword-bridge`: exposing Robot Framework keywords to agents as tools. This covers the meta tools, argument and variable handling, visibility and the default deny list, execution on Robot's main thread, and failure handling.

### Modified Capabilities

None. There are no existing specs yet.

## Impact

- New bridge module in `packages/slim`; its location is decided in design. Wiring in `runner.py` and the new options in `agent.py`.
- It reads keyword metadata from Robot's running namespace and libdoc. These are partly internal APIs, so compatibility across Robot Framework 7.x needs tests.
- Depends on `add-robot-tool-context` (main-thread execution). Benefits from `rework-tool-registry` (consistent name resolution).
- Adds a guide and a runnable example suite (see `rework-examples-docs`). It supersedes the planner/executor and free-text-spec snippets in `docs/examples`.
- Enables `add-agent-step-export`.
