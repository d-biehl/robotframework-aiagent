## v0.4.1 (2026-11-18)

### Refactor

- **agent**: wait for the stop signal instead of polling
- share one helper for reading Literal values
- **agent**: drop the unused hybrid keyword machinery

## v0.4.0 (2026-11-17)

### BREAKING CHANGE

- Python 3.10 is no longer supported, the vertexai extra
is gone and model names need a provider prefix (e.g. openai:gpt-5-mini).
The openai: prefix now uses the OpenAI Responses API.

### Feat

- **agent**: log tool calls, usage and images in a readable way
- **content**: keep loaded files short in Robot's argument log
- require Python 3.11 and upgrade to pydantic-ai v2
- **verify**: add LLM-as-judge verification package

### Fix

- **agent**: log a text answer only once
- **examples**: keep files the Playwright agent saves in the output dir
- **verify**: drop the implicit temperature default
- **agent**: keep pydantic-ai's first-run banner out of the console
- **examples**: pass the requested image size to the API
- **examples**: generate images with gpt-image-2
- **examples**: start a fresh Playwright MCP server per test
- **agent**: log output tool calls with pydantic-ai v2
- **examples**: create the OpenAI client lazily in ImageTool
- **agent**: do not swallow exceptions from failed agent runs

### Refactor

- **agent**: leave the estimated cost out of the usage line
- **agent**: extract AgentRunner core and migrate to pydantic-ai capabilities API

## v0.3.0 (2026-02-09)

### Feat

- implement some funtion to load content to an agent
- add OpenRouter model and utility functions for provider handling
- add logging for agent node processing and tool calls

### Fix

- improve formatting in MCPServerStdioEx and update image generation test settings

## v0.2.0 (2026-02-06)

### Feat

- implement OpenAI model handling

## v0.1.0 (2026-02-05)

### Feat

- add commitizen configuration for conventional commits
- update dependencies for robotframework-aiagent-slim to include openrouter

### Fix

- change library scope from 'SUITE' to 'TEST'

## v0.0.1 (2025-10-01)

### Feat

- **examples**: add image generation tool with OpenAI integration and update weather tools documentation
- add tool and toolset support with plugins
- add examples and tests for AIAgent functionality, including weather tools and invoice processing

### Fix

- refine agent event loop handling and improve error cleanup

### Refactor

- simplify type alias resolution and improve readability
- correct some spelling errors
