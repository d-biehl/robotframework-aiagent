# Proposal

## Why

With the keyword bridge, an agent can explore a system through Robot keywords. Once an exploration succeeds, teams want to freeze it as a deterministic test that is cheap, fast and repeatable without a model. The bridge already knows which keywords ran, with which arguments, results and status. Turning that record into Robot Framework code closes the loop from "explore with AI" to "run without AI".

## What Changes

- The keyword bridge records every keyword call it executes: keyword name, positional and named arguments, assignment, status and failure message.
- A new keyword (name decided in design) returns the recorded steps as Robot Framework code: keyword call lines, with `${name}=` for values the bridge assigned. It can optionally write them to a file.
- Options decided in design:
  - the scope of the record: the last `Chat` or the whole test
  - whether failed attempts and retried calls are included or dropped
  - output formatting (separator, indentation)
  - whether a minimal `*** Settings ***` section with the used libraries and resources is emitted
- Arguments that contain values from variables the agent must not disclose, and return values that are objects, are referenced through variables, never inlined. The exact rules are decided in design.

## Capabilities

### New Capabilities

- `agent-step-export`: recording keyword calls the agent executes through the keyword bridge and exporting them as Robot Framework code.

### Modified Capabilities

None. `keyword-bridge` does not exist as a spec yet. If it has been archived by the time this change is specified, the recording requirement may become a modification of `keyword-bridge` instead.

## Impact

- Keyword bridge module: recording of executed calls.
- `packages/slim/src/AIAgent/Agent/agent.py`: the new export keyword.
- A guide and a runnable example suite (see `rework-examples-docs`).
- Depends on `add-keyword-bridge`.
