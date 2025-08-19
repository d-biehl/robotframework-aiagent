*** Settings ***
Library         AIAgent    gpt-5-mini    name=RobotAgent


*** Test Cases ***
Who are you
    Chat    Who are you?
    Chat    What's your name?