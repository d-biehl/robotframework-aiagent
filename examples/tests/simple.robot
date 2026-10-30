*** Settings ***
Library     AIAgent.Agent    openai:gpt-5-mini


*** Test Cases ***
Who are you
    Chat    Who are you?
    Chat    What's your name?
