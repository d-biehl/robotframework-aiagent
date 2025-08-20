# Structured classification

Return a typed verdict with reasoning.

```robot
*** Settings ***
Library    AIAgent
...        output_type=${{ dataclasses.make_dataclass('Verdict', [('ok', bool), ('reason', str)]) }}
...        AS    QAJudge

*** Test Cases ***
Assess Answer
	${v}    QAJudge.Chat
	...    Evaluate whether the following text meets the acceptance criteria:
	...    - Must mention an order number
	...    - Must not contain personally identifiable information
	...    Text: "Your order #A123 has shipped."
	Log    OK=${v.ok}  REASON=${v.reason}
```
