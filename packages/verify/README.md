# robotframework-aiagent-verify

LLM-as-judge verification keywords for [robotframework-aiagent](https://github.com/d-biehl/robotframework-aiagent).

Provides the `AIAgent.Verify` Robot Framework library: ask a model to judge whether
some subject satisfies a set of criteria and turn the answer into a stable, typed
test assertion.

```robot
*** Settings ***
Library    AIAgent.Verify    openai:gpt-5-mini

*** Test Cases ***
Invoice Has Total
    ${response}=    Some Keyword That Produces Text
    # Fail-fast assertion: raises (and logs rationale + violations) if not satisfied
    Verify That    ${response}    nennt einen Gesamtpreis in EUR

    # Inspect without failing: returns a Verdict object you can assert on yourself
    ${verdict}=    Get Verdict    ${response}    ist höflich formuliert
    Should Be True    ${verdict.confidence} > 0.8
```

A `Verdict` carries `passed`, `confidence`, `rationale` and `violations`.

For more stable verdicts with a model that accepts sampling settings, pass them when
importing the library, e.g. `model_settings={'temperature': 0}`. Reasoning models such
as `gpt-5` do not accept them.

This is an optional plugin built on top of `robotframework-aiagent-slim`; install the
slim package with the provider extras you need (e.g. `[openai]`).
