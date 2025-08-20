# Quickstart

Minimal hello and per-step model override.

```robot
*** Settings ***
Library    AIAgent    gpt-5-chat-latest

*** Test Cases ***
Say Hello
	AIAgent.Chat    Hello, I am a Robot Framework test.
	AIAgent.Chat    What can you do?    model=google-gla:gemini-2.5-flash-lite
```
