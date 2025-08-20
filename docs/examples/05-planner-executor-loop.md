# Planner → Executor loop

Agent proposes steps, executor executes.

```robot
*** Settings ***
Library    AIAgent    AS    Planner
Library    AIAgent    AS    Executor

*** Test Cases ***
Plan And Do (Loop)
	${Plan}    Set Variable    ${{ dataclasses.make_dataclass('Plan', [('action', str), ('arg', str)]) }}
	FOR    ${i}    IN RANGE    3
		${step}    Planner.Chat
		...    Propose the next action as a tuple (action, arg). Keep it simple.
		...    Examples: Click(#submit), Assert("Logged in").
		...    Return only the action and arg.
		...    output_type=${Plan}

		${result}    Executor.Chat
		...    Execute: ${step.action} ${step.arg}
		...    Respond with a brief status line.

		# Optionally break on success keyword in ${result}
	END
```
