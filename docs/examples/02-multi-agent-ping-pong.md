# Multi-agent ping pong

Two agents talking to each other via aliases.

```robot
*** Settings ***
Library    AIAgent    openai:gpt-5-chat-latest
Library    AIAgent    anthropic:claude-sonnet-4-0    AS    SecondAgent

*** Test Cases ***
Ping Pong
	${q}    AIAgent.Chat      Ask me one question.
	${a}    SecondAgent.Chat  ${q}
	AIAgent.Chat               ${a}
```
