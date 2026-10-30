# Per-step model switch

Two turns, different models.

```robot
*** Settings ***
Library    AIAgent

*** Test Cases ***
Per-Step Model Switch
	AIAgent.Chat    Summarize briefly.    model=anthropic:claude-sonnet-4-0
	AIAgent.Chat    Now translate to German.    model=google:gemini-2.5-flash-lite
```
