# Config by example

Import-time defaults and overrides.

```robot
*** Settings ***
Library    AIAgent
...        model=gpt-5-chat-latest
...        instructions=You are a helpful QA assistant.
...        system_prompt=Answer concisely.
...        output_type=${{ str }}
...        retries=1
...        output_retries=1

*** Test Cases ***
Config Smoke (Example Only)
	# This file demonstrates import-time configuration.
	# Add your own steps here as needed.
	No Operation
```
