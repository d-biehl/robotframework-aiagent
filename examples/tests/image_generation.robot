*** Settings ***
Library     AIAgent    gpt-5-chat-latest

*** Test Cases ***
Simple Image with gpt-5-chat
    AIAgent.Chat    Welche Tools hast Du?
    AIAgent.Chat    Male mir ein Bild von einem Käguruh das mit einem Roboter boxt. Es soll im Futuristisch Sci-Fi Stil sein.
    ...    frag nicht weiter nach. Male einfach
