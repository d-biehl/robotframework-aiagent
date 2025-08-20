# Structured outputs (default)

Default typed schema at import; simple call.

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
