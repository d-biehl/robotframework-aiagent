*** Settings ***
Library    AIAgent.Agent    gpt-5    name=RobotAgent   instructions=${INSTRUCTIONS}  toolsets=["playwright", "weather_toolset", "datetime_toolset"]    retries=3    output_retries=3
Variables       AIAgent.Examples.PlaywrightMCP
#Library         AIAgent.Agent    gpt-5-mini    instructions=${INSTRUCTIONS}    toolsets=${{[$playwright]}}
Variables    pydantic_ai.models.openai
Variables    pydantic_ai.providers.openrouter
Variables       pydantic_ai.providers.ollama
# Library    AIAgent.Agent    ${{$OpenAIModel("google/gemini-2.5-pro", provider=$OpenRouterProvider())}}    toolsets=${{[$playwright]}}
# Library    AIAgent.Agent    ${{$OpenAIModel("google/gemini-2.5-pro", )}}    toolsets=${{[$playwright]}}
# Library    AIAgent.Agent    ${{$OpenAIModel("qwen/qwen3-4b-2507", provider=$OllamaProvider(base_url="http://192.168.178.103:1234/v1"))}}    toolsets=${{[$playwright]}}
# Library    AIAgent.Agent    ${{$OpenAIModel("openai/gpt-oss-20b", provider=$OllamaProvider(base_url="http://localhost:1234/v1"))}}    instructions=${INSTRUCTIONS}    toolsets=${{[$playwright, "datetime_toolset", "weather_toolset"]}}    model_settings={"max_tokens": 131072, "temperature": 0.1}


*** Variables ***
${INSTRUCTIONS}
...                 You are a System Test Agent. Complete the user's request using the available tools with minimal, safe, and deterministic actions.
...                 ${EMPTY}
...                 Tool use:
...                 - For each user message, use the fewest tools needed; prefer the simplest single tool that can complete the task.
...                 - Avoid exploratory or redundant calls.
...                 - Report only what tools return or what you have directly verified; do not fabricate results.
...                 ${EMPTY}
...                 Web tasks:
...                 - Take deterministic steps only (e.g., navigate, handle consent/cookies).
...                 - Stop immediately once the goal is achieved.
...                 ${EMPTY}
...                 Output:
...                 - On success: "Success: <brief fact>".
...                 - On failure: "Failed: <brief reason>".
...                 - Be concise. No extra commentary, code, or screenshots unless requested.
...                 - Reply in the user's language.
...                 - Do not ask what to do next.
...                 ${EMPTY}
...                 Context:
...                 - Use only the message history within this test.
...                 ${EMPTY}
...                 Safety:
...                 - Avoid irreversible actions, authentication, or sensitive operations unless explicitly requested and clearly safe.
...                 - Never enter credentials, make purchases, delete, or change user settings/data.
...                 - If a requested action requires authentication or appears risky, stop and report why.
...                 - If a needed tool is unavailable, report the limitation and stop.


*** Test Cases ***
check heise news
    Chat    Welche Tools hast Du?
    Chat    Navigiere zu https://heise.de
    Chat    Stimme dem Consent Banner zu, falls vorhanden
    Chat    Gib mir die neuesten Nachrichten

check bild news
    Chat    Welche Tools hast Du?
    Chat    Navigiere zu https://bild.de
    Chat    Stimme dem Consent Banner zu, falls vorhanden
    Chat    Gib mir die neuesten Nachrichten
