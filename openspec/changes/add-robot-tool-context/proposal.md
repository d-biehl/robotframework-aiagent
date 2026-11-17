# Proposal

## Why

Tools run in the agent's worker thread or in pydantic-ai's thread pool. Robot Framework drops every log message that does not come from `MainThread` (`LOGGING_THREADS` in `robot.output.librarylogger`), so the log lines written by tools, such as those in the ImageTool example, silently disappear. `BuiltIn()` calls from those threads only work by accident, and tool authors have no supported way to reach Robot at all.

The planned keyword bridge has the same root problem. Keywords must run in Robot's thread. Robot also only runs async keywords natively when no event loop is running in that thread. Today `Chat` is an async keyword, so a loop is always running while the agent works.

## What Changes

- **BREAKING**: `Chat` and the `Verify` keywords become synchronous keywords. While the agent runs in its worker thread, `Chat` services a work queue in `MainThread` until the run finishes.
  - All interaction with Robot (logging, variable access and, later, keyword execution) goes through this queue.
  - Work therefore runs in `MainThread` with no event loop running, and Robot executes async keywords natively.
- **BREAKING**: `AgentRunner.run` becomes synchronous. Sibling libraries built on the runner (`Verify`) are adapted.
- Run logging moves from callbacks on Robot's event loop to the work queue.
- A new `RobotContext` gives tools:
  - logging into the Robot log (level, HTML)
  - Robot's output directory
  - read access to Robot variables
  - an internal primitive to execute a callable in `MainThread` and wait for its result, which the keyword bridge will build on
- Tools reach the context in two ways:
  - an accessor, `robot_context()`, which works in plain functions from variable files, sync or async, without changing their signature
  - pydantic-ai deps, `RunContext[RobotContext]`, for tools that want it typed. The agent's deps type becomes `RobotContext`.
- Robot test timeouts and the `timeout` argument of `Chat` still interrupt the wait. A timed-out or failed run cancels the agent task.
- Outside a Robot run (unit tests, plain Python), `RobotContext` degrades gracefully: logging falls back to Python logging and there is no output directory.
- The ImageTool example uses `RobotContext` for its output directory and logging.

## Capabilities

### New Capabilities

- `robot-context`: how agent runs interact with Robot Framework. Tools get thread-safe logging into the Robot log, the output directory and variable access, through both the accessor and deps. All Robot interaction during `Chat` happens on Robot's main thread, without a running event loop.

### Modified Capabilities

None. There are no existing specs yet.

## Impact

- `packages/slim/src/AIAgent/Agent/runner.py`: execution model, work queue, logging.
- `packages/slim/src/AIAgent/Agent/agent.py`: `Chat` becomes synchronous.
- `packages/verify/src/AIAgent/Verify/verify.py`: keywords become synchronous.
- New public module for `RobotContext` and `robot_context()`. Its import path is decided in design.
- `examples/src/AIAgent/Examples/ImageTool.py`.
- Python code that awaited `Chat` or `AgentRunner.run` directly must call them synchronously.
- Foundation for `rework-tool-registry` (factories use the context) and `add-keyword-bridge` (keyword execution).
