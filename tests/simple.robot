*** Settings ***
Variables       AIAgent.Models.OpenAI
Library         AIAgent.Agent    ${OpenAIResponsesModel("${OPENAI_MODEL_GPT_5_2_PRO}")}
...                 toolsets=playwright
...                 model_settings=${OPENAI_MODEL_SETTINGS}
# Library    AIAgent.Agent    anthropic:claude-sonnet-4-5
# ...    toolsets=playwright
# ...    model_settings=${OPENAI_MODEL_SETTINGS}


*** Variables ***
&{OPENAI_MODEL_SETTINGS}
...                         openai_reasoning_effort=high
...                         openai_response_format=verbose
...                         openai_reasoning_summary=detailed
...                         openai_chat_send_back_thinking_parts=field


*** Test Cases ***
simple hello
    Chat    Hello, who are you? And who am I?

second
    Chat    Wie ist das Wetter in Berlin heute?
