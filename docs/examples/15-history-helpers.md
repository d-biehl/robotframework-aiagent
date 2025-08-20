# History helpers

Retrieve complete and delta message histories; JSON option.

```robot
*** Settings ***
Library    AIAgent

*** Test Cases ***
History Helpers
	# Produce a tiny bit of history first
	AIAgent.Chat    Hello, please remember this line.

	# Full history of the last completed run
	${history}    AIAgent.Get Complete History
	Log Many      ${history}

	# Only the new messages from the most recent run (since prior)
	${delta}      AIAgent.Get New History
	Log Many      ${delta}

	# Return JSON-encoded messages instead of raw structures
	${json}       AIAgent.Get Complete History    format=JSON
	Log           ${json}
```
