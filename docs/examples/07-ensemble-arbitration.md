# Ensemble + Arbitration

Two answers; an arbiter chooses the best.

```robot
*** Settings ***
Library    AIAgent    gpt-5-nano          AS    A1
Library    AIAgent    claude-sonnet-4-0   AS    A2
Library    AIAgent    google-flash-2-4
...        output_type=${{ dataclasses.make_dataclass('Choice', [('best', int), ('reason', str)]) }}
...        AS    Arbiter

*** Test Cases ***
Pick Best Answer
	${q}     Set Variable    Explain the difference between mocks and stubs.
	${r1}    A1.Chat    ${q}
	${r2}    A2.Chat    ${q}
	${c}     Arbiter.Chat
	...    Choose best=1 for A1 or best=2 for A2.
	...    Consider correctness and clarity. Return best and reason.
	Log     Best=${c.best}  Reason=${c.reason}
```
