# Fallback routing

Start fast/cheap; escalate if not confident.

```robot
*** Settings ***
Library    AIAgent    gpt-5-nano    AS    Fast
Library    AIAgent                 AS    Strong

*** Test Cases ***
Cheap Then Strong
	${Verdict}    Set Variable    ${{ dataclasses.make_dataclass('Verdict', [('confident', bool), ('answer', str)]) }}
	${v}    Fast.Chat    Answer briefly. Return confident + answer.    output_type=${Verdict}
	IF    not ${v.confident}
		${answer}    Strong.Chat    Provide a precise, verified answer.
	ELSE
		${answer}    Set Variable    ${v.answer}
	END
	Log    ${answer}
```
