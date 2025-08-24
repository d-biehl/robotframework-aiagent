*** Settings ***
Library     AIAgent.Agent


*** Test Cases ***
Say Hello
    # AIAgent.Chat    Hello, I am a Robot Framework test.
    Chat    What can you do?    model=gpt-5-nano
    ${messages}    Get Message History
    Log    ${messages}

    ${messages}    Get Message History    JSON
    Log    ${messages}
