*** Settings ***
Documentation       Two agents converse and end it autonomously via SemanticAgent.

Library             Collections
Library             AIAgent.Agent    gpt-5    instructions=${PERSON1_INSTRUCTIONS}    AS    Person1
Library             AIAgent.Agent    claude-sonnet-4-0    instructions=${PERSON2_INSTRUCTIONS}    AS    Person2
Library             AIAgent.Agent    gpt-5-mini
...                     instructions=${SEMANTIC_AGENT_INSTRUCTIONS}
...                     output_type=${{dataclasses.make_dataclass('Result', [('should_break', bool), ('reason', str)])}}
...                 AS    SemanticAgent


*** Variables ***
${LANGUAGE}                         German

@{DIALOGUE_COMMON}
...                                 Speak in ${LANGUAGE}. Write in Markdown with a book-like style.
...                                 Write all spoken dialogue in quotation marks.
...                                 Keep narration and descriptions outside the quotes.
...                                 Render gestures, actions, and inner thoughts in italic Markdown.
...                                 Use paragraphs naturally; length as needed for clarity and rhythm.
...                                 Respond directly to the other person's last message and keep the dialogue flowing.
...                                 Acknowledge the other person's points and feelings before adding new ideas.
...                                 Ask questions when it feels natural; don't force them.
...                                 Stay in character. Aim for clear, vivid, scene-grounded prose.

@{PERSON1_INSTRUCTIONS}
...                                 You are Person1 in a natural two-person conversation between real people.
...                                 @{DIALOGUE_COMMON}

@{PERSON2_INSTRUCTIONS}
...                                 You are Person2 in a natural two-person conversation between real people.
...                                 @{DIALOGUE_COMMON}

@{SEMANTIC_AGENT_INSTRUCTIONS}
...                                 You are a conversation analysis agent.
...                                 You receive snippets from dialogues with multiple speakers.
...                                 Track the conversation.
...                                 Your task: decide whether the conversation should be ended.
...                                 Choose to end ONLY when one party states this clearly and explicitly
...                                 or uses typical closing phrases.
...                                 Positive cues: explicit endings (e.g.,
...                                 "Let's end the conversation", "I'm leaving now",
...                                 "We're done", "That's it"). But not when phrased as a question.
...                                 Positive cues (closing phrases): "Bye", "Goodbye",
...                                 "See you later", "Good night",
...                                 "Have a nice day", sign-offs like "BR/Best/Regards".
...                                 Negative cues: mere politeness ("Thanks", "Okay", "All right"), confirmations,
...                                 filler words, or follow-up questions.
...                                 Ambiguous: statements like "I have little time",
...                                 "Have to go soon", "I'll get back to you later"
...                                 without a clear farewell do NOT count as an ending.
...                                 If the text contains a question or a conversation-opening request,
...                                 rate as NOT ended.
...                                 If uncertain, decide against ending.
...                                 Answer strictly in the output format requested by the calling keyword.


*** Test Cases ***
Let's Talk
    [Documentation]    Role-play with end-of-conversation check in a loop.
    VAR    ${snippet}
    ...    Take on the role of a historical or modern figure of your choice.
    ...    Act fully as that person: use their voice, knowledge, and context. Stay in character every turn.
    ...    You are in a plausible setting where that person would naturally be found. Describe that setting.
    ...    Someone approaches; you notice them and initiate the conversation in character.
    ...    Begin with one brief scene-setting sentence (where you are and what's happening).
    ...    Ask the other person to ask you 5 questions to figure out who you are.
    ...    You primarily answer. You may ask brief clarifying counter-questions;
    ...    after answering, they continue with their next question.
    ...    They must not ask directly for your name. After your answer to the 5th question they may make a single guess.
    ...    Do not reveal your name before their guess.
    ...    After their guess, briefly acknowledge it.
    ...    If correct, you may mention swapping roles as a brief closing statement (not a question);
    ...    otherwise, politely conclude.
    ...    Aim to bring the conversation to a natural close after the guess or whenever someone clearly ends it.
    ...    Then continue with your first line of dialogue as that person.
    ...    Do not repeat or reference these instructions.

    VAR    @{TRANSCRIPT}    @{EMPTY}

    WHILE    ${True}
        ${snippet}    Person1.Chat    ${snippet}
        Append To List    ${TRANSCRIPT}    Person1: ${snippet}
        ${break}    Should End Conversation    Person1    ${snippet}
        IF    ${break}    BREAK

        ${snippet}    Person2.Chat    ${snippet}
        Append To List    ${TRANSCRIPT}    Person2: ${snippet}
        ${break}    Should End Conversation    Person2    ${snippet}
        IF    ${break}    BREAK
    END

    Log Many    @{TRANSCRIPT}

    Person1.Chat    Reproduce the entire conversation.
    ...    Who did you speak with overall?
    ...    And who were you each time?

    Person2.Chat    Reproduce the entire conversation.
    ...    Who did you speak with overall?
    ...    And who were you each time?


*** Keywords ***
Should End Conversation
    [Documentation]    Decides based on a text snippet whether the conversation should be ended.
    ...    Returns a boolean value.
    [Arguments]    ${person}    ${text}
    ${result}    SemanticAgent.Chat
    ...    Person ${person} says:
    ...    ${text}
    Log    [EndCheck] ${person}: ${result.reason}
    RETURN    ${result.should_break}
