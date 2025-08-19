*** Settings ***
Library     AIAgent    gpt-5-chat-latest


*** Test Cases ***
First
    [Documentation]    Kurzer Begrüßungsdialog.
    AIAgent.Chat    Hallo, ich bin Tobor! Mit wem spreche ich?
    AIAgent.Chat    Was kannst Du?    model=google-gla:gemini-2.5-flash-lite
    AIAgent.Chat    Wer bin ich? Und wer bist Du?    model=claude-sonnet-4-0
