*** Settings ***
Library     AIAgent.Agent    openai:gpt-5.4-mini


*** Test Cases ***
First
    [Documentation]    Kurzer Begrüßungsdialog.
    Chat    Hallo, ich bin Tobor! Mit wem spreche ich?
    Chat    Was kannst Du?    model=openrouter:google/gemini-2.5-flash-lite
    Chat    Wer bin ich? Und wer bist Du?    model=anthropic:claude-sonnet-5-5
