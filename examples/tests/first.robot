*** Settings ***
Library     AIAgent.Agent    gpt-5-chat-latest


*** Test Cases ***
First
    [Documentation]    Kurzer Begrüßungsdialog.
    Chat    Hallo, ich bin Tobor! Mit wem spreche ich?
    Chat    Was kannst Du?    model=google-gla:gemini-2.5-flash-lite
    Chat    Wer bin ich? Und wer bist Du?    model=claude-sonnet-4-0
