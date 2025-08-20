# Author ↔ Reviewer

Draft and review with typed edits.

```robot
*** Settings ***
Library    AIAgent    AS    Author
Library    AIAgent
...        output_type=${{ dataclasses.make_dataclass('Review', [('ok', bool), ('edits', str), ('reason', str)]) }}
...        AS    Reviewer

*** Test Cases ***
Draft And Review
	${draft}    Author.Chat    Write a 3-sentence product update note.
	${r}        Reviewer.Chat  Review for clarity and tone. If not ok, propose edits.    ${draft}
	IF    not ${r.ok}
		${draft}    Author.Chat    Apply these edits: ${r.edits}
	END
```
