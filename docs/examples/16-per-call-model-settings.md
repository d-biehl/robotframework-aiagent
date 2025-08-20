# Per-call model settings

Demonstrate model_settings override for a step.

```robot
*** Settings ***
Library    AIAgent

*** Test Cases ***
Per-Call Model Settings
	# Tweak model settings for a single step
	${settings}    Set Variable    ${{ {'temperature': 0.2, 'max_output_tokens': 256} }}
	AIAgent.Chat    Summarize this text in one sentence.    model_settings=${settings}
```
