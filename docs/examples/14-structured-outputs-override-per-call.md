# Structured outputs (override per call)

Inline dataclass schema override example.

```robot
*** Settings ***
Library    AIAgent

*** Test Cases ***
Structured Output Override Per Call
	${schema}    Set Variable    ${{ dataclasses.make_dataclass('Answer', [('value', int)]) }}
	${answer}    AIAgent.Chat    Give me a number between 1 and 10.    output_type=${schema}
	Log    ${answer.value}
```
