# Multimodal attachment (placeholder)

Model/provider-dependent example.

```robot
*** Settings ***
Library    AIAgent

*** Test Cases ***
Multimodal Attachment (Model-Dependent)
	# Pseudocode-style; exact attachment syntax depends on model/provider support
	${img}      Get File    screenshots/login.png
	${summary}  AIAgent.Chat    Analyze this screenshot for missing labels and contrast issues.
	...                          attachments=${img}
	Log         ${summary}
```
