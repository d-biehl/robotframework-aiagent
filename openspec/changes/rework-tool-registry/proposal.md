# Proposal

## Why

`PluginManager` mirrors Robot's rules for static variable files (`__all__`, leading underscores), and that is the right invariant. A plugin module is a valid Robot variable file, so the same module works through an entry point and through `Variables`. That matters because many projects are not packaged with a project manager like uv and can only use variable files. Around that invariant, several things are missing or fragile:

- A plugin that fails to import breaks the import of `AIAgent.Agent` as a whole.
- Name collisions between plugins overwrite each other silently.
- Dynamic variable files (`get_variables()`) work in Robot but not as plugins, so the two paths diverge.
- Toolset instances are module globals shared by every agent and every test. MCP toolsets need a `keep_alive=False` workaround. Two agents using the same MCP toolset in one test risk sharing one session across two event loops.
- Capabilities cannot be referenced by name, only through `${{ }}` expressions, although pydantic-ai ships many ready-made ones.
- `Chat` accepts toolset objects per call, but not toolset names the way the library import does.

## What Changes

- Make the invariant explicit. Every plugin module is a valid Robot variable file, and the names the registry resolves equal the variable names Robot creates from that module.
- Isolate plugin failures. A plugin that fails to load is reported as a warning and skipped. Name collisions are reported, with a defined resolution order.
- Support `get_variables()` modules as plugins, as Robot does for variable files.
- Make capabilities a third kind next to tools and toolsets:
  - Module attributes that are pydantic-ai capabilities are registered under their name.
  - The library's `capabilities` argument accepts names.
- Ship `AIAgent.Capabilities` with slim. It exposes pydantic-ai's built-in capabilities (for example web search, web fetch, thinking and image generation) under stable names, both as a registered plugin and as a variable file.
- Let `Chat` accept toolset names per call, like the library import.
- Add a `@toolset_factory` decorator that marks a callable as a factory instead of a tool:
  - `scope='test'` (default) creates a fresh instance for every test.
  - `scope='suite'` creates one instance per suite, shared by that suite's agents.
  - Factories work the same way through entry points and through variable files. Undecorated callables stay tools, as today.
- Suite-scoped instances require an agent event loop that outlives a single test. Its lifecycle is decided in design.
- Adapt the examples:
  - The Playwright MCP toolset becomes a factory that resolves its output directory at creation time through the Robot context, and the `keep_alive` workaround goes away.
  - The ImageTool example is replaced by the shipped image-generation capability.

## Capabilities

### New Capabilities

- `tool-registry`: discovery and resolution of tools, toolsets and capabilities from entry-point plugins and Robot variable files. This covers the variable-file invariant, failure isolation, name resolution for the library import and per `Chat` call, and factories with test or suite lifetime.

### Modified Capabilities

None. There are no existing specs yet.

## Impact

- `packages/slim/src/AIAgent/Agent/plugin_manager.py`: loading, isolation, `get_variables()`, capabilities, factories.
- `packages/slim/src/AIAgent/Agent/runner.py` and `agent.py`: name resolution, per-call names, event loop lifetime for suite-scoped instances.
- New `packages/slim/src/AIAgent/Capabilities` and an entry point registered by slim.
- `examples/`: `PlaywrightMCP.py`, `ImageTool.py`, `pyproject.toml` entry points, and the suites `wheater.robot`, `playground.robot` and `image_generation.robot`.
- Depends on `add-robot-tool-context`, which provides the context for factories and the work queue that the loop lifecycle builds on.
