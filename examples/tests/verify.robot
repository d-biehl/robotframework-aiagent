*** Settings ***
Library    AIAgent.Verify    gpt-5-mini


*** Test Cases ***
Verify That Passes
    Verify That    The invoice total is 199.00 EUR.    nennt einen Gesamtpreis in EUR

Verify That Fails
    [Documentation]    Demonstrates a failing verification (expected to fail).
    Verify That    The weather is nice today.    nennt einen Gesamtpreis in EUR

Inspect A Verdict Without Failing
    ${verdict}=    Get Verdict    Hallo, schön dich zu sehen!    ist eine höfliche Begrüßung
    Log    Rationale: ${verdict.rationale}
    Should Be True    ${verdict.passed}
    Should Be True    ${verdict.confidence} > 0.5
