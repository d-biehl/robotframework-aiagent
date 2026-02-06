*** Settings ***
Variables  AIAgent.Models.OpenAI
Library  AIAgent.Agent    ${OpenAIResponsesModel("${OPENAI_MODEL_GPT_5_2_PRO}")}
...    model_settings={"openai_reasoning_effort": "high", "openai_response_format": "verbose"}


*** Test Cases ***
simple hello
    Chat    Hello, who are you?