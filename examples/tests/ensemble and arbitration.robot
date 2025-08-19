*** Settings ***
Documentation       Demonstrates the "Ensemble and Arbitration" pattern.
...                 Two agents (Parser1, Parser2) extract a typed Invoice from the same text,
...                 and a third agent (Arbiter) selects the better candidate using simple
...                 checks (net + VAT ≈ gross, ISO date, EU VAT-ID regex).
...                 Also shows:
...                 - Using output_type with inline dataclasses for typed results
...                 - Mixing providers per agent
...                 - Reading a test resource with ${CURDIR}

Library             OperatingSystem
Library             AIAgent    openai:gpt-5-nano    AS    Parser1
Library             AIAgent    openai:gpt-4.1-nano    AS    Parser2
Library             AIAgent    openai:gpt-5
...                     output_type=${{ dataclasses.make_dataclass('Choice', [('best', int), ('reason', str)]) }}
...                 AS    Arbiter


*** Variables ***
${InvoiceFields}    ${{ [('number', str), ('date', str), ('net', float), ('vat', float), ('gross', float), ('vat_id', str)] }}
${Invoice}          ${{ dataclasses.make_dataclass('Invoice', $InvoiceFields) }}


*** Test Cases ***
Invoice Ensemble
    [Documentation]    Flow:
    ...    (1) Parser1 & Parser2 return Invoice(number, date, net, vat, gross, vat_id).
    ...    (2) Arbiter returns Choice(best, reason) after validating arithmetic,
    ...    date format (YYYY-MM-DD) and EU VAT pattern.
    ...    (3) We pick the winner and assert VAT regex and net+vat≈gross.

    ${text}    Get File    ${CURDIR}/resources/invoice_123.txt

    ${r1}    Parser1.Chat
    ...    Extract invoice fields from the text. Return Invoice(number, date, net, vat, gross, vat_id).
    ...    ${text}
    ...    output_type=${Invoice}

    ${r2}    Parser2.Chat
    ...    Extract invoice fields from the text. Return the same structure.
    ...    ${text}
    ...    output_type=${Invoice}

    ${c}    Arbiter.Chat
    ...    Choose best=1 for Parser1 or best=2 for Parser2.
    ...    Validate: net + vat == gross (±0.01), date in YYYY-MM-DD, vat_id matches EU VAT pattern.
    ...    Return best and a short reason. Use both candidates:
    ...    Candidate1=${r1} Candidate2=${r2}

    Log    Best=${c.best} Reason=${c.reason}

    IF    ${c.best} == 1
        VAR    ${final}    ${r1}
    ELSE
        VAR    ${final}    ${r2}
    END

    # Example assertions
    Should Match Regexp    ${final.vat_id}    ^[A-Z]{2}[A-Z0-9]{8,12}$
    Should Be True    ${{abs($final.net+$final.vat-$final.gross) < 0.01}}
