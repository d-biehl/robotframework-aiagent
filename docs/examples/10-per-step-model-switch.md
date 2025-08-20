# Per-step model switch

Two turns, different models.

```robot
*** Settings ***
Library    AIAgent

*** Test Cases ***
Per-Step Model Switch
	AIAgent.Chat    Summarize briefly.    model=claude-sonnet-4-0
	AIAgent.Chat    Now translate to German.    model=google-gla:gemini-2.5-flash-lite
```
