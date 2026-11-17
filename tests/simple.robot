*** Settings ***
Variables       AIAgent.Models.OpenAI
Variables       AIAgent.Models.OpenRouter
Variables       AIAgent.Content
Library         AIAgent.Agent    openrouter:anthropic/claude-sonnet-5.5
# Library         AIAgent.Agent    ${OpenAIResponsesModel("${OPENAI_MODEL_GPT_5_2_PRO}")}
# ...                 model_settings=${OPENAI_MODEL_SETTINGS}
# Library       AIAgent.Agent    ${OpenRouterModel("z-ai/glm-4.7")}
# Library       AIAgent.Agent    anthropic:claude-sonnet-4-5
# Library       AIAgent.Agent    github:openai/gpt-4.1
# Library       AIAgent.Agent    ollama:glm-4.7-flash:latest
# ...                 toolsets=${{["weather_toolset", "playwright"]}}


*** Variables ***
&{OPENAI_MODEL_SETTINGS}
...                         openai_response_format=verbose
...                         openai_reasoning_summary=detailed
...                         openai_chat_send_back_thinking_parts=field


*** Test Cases ***
simple hello
    Chat    Hello, who are you? And who am I?
    Chat    Wich tools do you have access to?

second
    Chat    What is the weather like in Berlin today?

third
    Chat    What is on this picture?
    ...    ${load_binary_content("${CURDIR}/invoice_example.png")}
    Chat    Is this a real invoice or a fake one?
    Chat    What is the total amount on this invoice?
    Chat    Are all legal details on this image correct?
    Chat    Does the total amount on this image match what is stated on the invoice?

fourth
    Chat    ${load_binary_content("${CURDIR}/invoice_example.png")}
    ...    What is the total amount on this invoice?
    ...    Are all legal details on this image correct?
    ...    Does the total amount on this image match what is stated on the invoice?
