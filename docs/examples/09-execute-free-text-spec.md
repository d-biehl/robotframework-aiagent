# Execute free-text spec

Guarded autonomy from a natural-language spec.

```robot
*** Settings ***
Library    AIAgent    AS    Planner

*** Test Cases ***
Execute Free-Text Spec
	${Spec}    Set Variable    As a user, open the demo app, log in as "alice", then verify greeting "Welcome, Alice".
	${Step}    Set Variable    ${{ dataclasses.make_dataclass('Step', [('keyword', str), ('args', list[str]), ('done', bool)]) }}

	WHILE    ${True}
		${s}    Planner.Chat
		...    Read the spec and propose the next Robot step as Step(keyword, args, done).
		...    Allowed keywords: Browser.Open, Browser.Fill Text, Browser.Click, Should Contain.
		...    Use only allowed keywords and valid args. No prose.
		...    output_type=${Step}

		IF    ${s.done}    BREAK
		Run Keyword    ${s.keyword}    @{s.args}

		# Optionally provide a short status back for planning continuity
		Planner.Chat    Executed: ${s.keyword}    message_history=${None}
	END
```
