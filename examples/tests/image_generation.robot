*** Settings ***
Library     AIAgent.Agent    openai:gpt-5.4-mini    toolsets=genimage_toolset

*** Test Cases ***
Simple Image with gpt-5.4-mini
    Chat    Welche Tools hast Du?
    Chat    Male mir ein Bild von einem Käguruh das mit einem Roboter boxt. Es soll im Futuristisch Sci-Fi Stil sein.
    ...    frag nicht weiter nach. Male einfach
