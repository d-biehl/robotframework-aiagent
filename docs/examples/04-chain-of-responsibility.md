# Chain of Responsibility

Extractor → Judge → Reporter with typed outputs.

```robot
*** Settings ***
Library    AIAgent
...        output_type=${{ dataclasses.make_dataclass('Data', [('order_id', str)]) }}
...        AS    Extractor
Library    AIAgent
...        output_type=${{ dataclasses.make_dataclass('Verdict', [('ok', bool), ('reason', str)]) }}
...        AS    Judge
Library    AIAgent    gpt-5-nano    AS    Reporter

*** Test Cases ***
Chain Of Responsibility
	${data}     Extractor.Chat    Extract the order id from: "Order #A123 has shipped to Berlin."
	${verdict}  Judge.Chat        Check if the text reveals PII. Return ok + reason.
	...                          The order id is ${data.order_id}.
	${summary}  Reporter.Chat
	...                          Summarize the outcome for a human reader based on:
	...                          - Order ID: ${data.order_id}
	...                          - Verdict OK: ${verdict.ok}
	...                          - Reason: ${verdict.reason}
	Log         ${summary}
```
