import importlib.metadata
from dataclasses import dataclass, field
from typing import Any

from AIAgent.Agent.runner import AgentRunner, KnownModelName
from pydantic_ai import models, settings
from robot.api import logger
from robot.api.deco import keyword, library
from robot.result.model import TestCase as ResultTestCase
from robot.running.model import TestCase as RunningTestCase

try:
    __version__ = importlib.metadata.version('robotframework-aiagent-verify')
except importlib.metadata.PackageNotFoundError:
    __version__ = '0.0.0'


@dataclass
class Verdict:
    """Structured result of a verification.

    Attributes:
        passed: True only if the subject satisfies every criterion.
        confidence: The judge's confidence in the verdict, from 0.0 to 1.0.
        rationale: A short explanation of the verdict.
        violations: The specific criteria that were not met (empty if passed).
    """

    passed: bool
    confidence: float
    rationale: str
    violations: list[str] = field(default_factory=list)


_DEFAULT_INSTRUCTIONS = """\
You are a strict, impartial evaluator inside an automated test.
You are given a SUBJECT and one or more CRITERIA.
Decide whether the SUBJECT satisfies ALL of the criteria.

Rules:
- passed is true only if every criterion is satisfied, otherwise false.
- confidence is your confidence in the verdict, from 0.0 to 1.0.
- rationale is one or two sentences explaining the verdict.
- violations lists the specific criteria that were not met (empty when passed).

Judge only against the given criteria. Do not invent additional requirements.
"""


@library(
    scope='TEST',
    version=__version__,
    auto_keywords=False,
    listener='SELF',
)
class Verify:
    """LLM-as-judge verification keywords for Robot Framework.

    Wraps a pydantic-ai agent (through the shared `AgentRunner` core) that is
    configured to return a structured `Verdict`. Use `Verify That` for fail-fast
    assertions and `Get Verdict` to inspect the verdict without failing the test.

    The judge defaults to deterministic decoding (temperature 0) unless you pass
    your own `model_settings`. Each verification is evaluated independently; no
    conversation history is shared between calls.
    """

    def __init__(
        self,
        model: models.Model | KnownModelName | str | None = None,
        *,
        instructions: str | None = None,
        model_settings: settings.ModelSettings | dict[str, Any] | None = None,
        confidence_threshold: float = 0.0,
        retries: int = 1,
    ):
        # Default to deterministic judging unless the caller overrides settings.
        resolved_settings: settings.ModelSettings | dict[str, Any]
        resolved_settings = {'temperature': 0.0} if model_settings is None else model_settings

        self._confidence_threshold = confidence_threshold
        self._runner: AgentRunner[Verdict] = AgentRunner(
            model,
            output_type=Verdict,
            instructions=instructions if instructions is not None else _DEFAULT_INSTRUCTIONS,
            model_settings=resolved_settings,
            retries=retries,
        )

    def end_test(self, data: RunningTestCase, result: ResultTestCase) -> None:
        self._runner.shutdown()

    @keyword
    async def verify_that(
        self,
        subject: object,
        *criteria: str,
        model: models.Model | KnownModelName | str | None = None,
        model_settings: settings.ModelSettings | dict[str, Any] | None = None,
        confidence_threshold: float | None = None,
        timeout: float | None = None,
    ) -> Verdict:
        """Assert that `subject` satisfies all `criteria`, failing the test if not.

        Logs the verdict (rationale and any violations) and raises an
        ``AssertionError`` when the verdict is not ``passed`` (or its confidence is
        below the configured threshold). Returns the `Verdict` on success.
        """
        verdict = await self._evaluate(subject, criteria, model=model, model_settings=model_settings, timeout=timeout)
        self._log_verdict(verdict)

        threshold = self._confidence_threshold if confidence_threshold is None else confidence_threshold
        if not (verdict.passed and verdict.confidence >= threshold):
            raise AssertionError(self._failure_message(verdict))
        return verdict

    @keyword
    async def get_verdict(
        self,
        subject: object,
        *criteria: str,
        model: models.Model | KnownModelName | str | None = None,
        model_settings: settings.ModelSettings | dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> Verdict:
        """Evaluate `subject` against all `criteria` and return the `Verdict`.

        Logs the verdict but never fails the test, so the caller can inspect
        ``passed``, ``confidence``, ``rationale`` and ``violations`` directly.
        """
        verdict = await self._evaluate(subject, criteria, model=model, model_settings=model_settings, timeout=timeout)
        self._log_verdict(verdict)
        return verdict

    async def _evaluate(
        self,
        subject: object,
        criteria: tuple[str, ...],
        *,
        model: models.Model | KnownModelName | str | None,
        model_settings: settings.ModelSettings | dict[str, Any] | None,
        timeout: float | None,
    ) -> Verdict:
        if not criteria:
            raise ValueError('Verify requires at least one criterion.')

        verdict = await self._runner.run(
            self._build_prompt(subject, criteria),
            message_history=None,  # each verification is independent
            model=model,
            model_settings=model_settings,
            timeout=timeout,
        )
        if verdict is None:
            raise RuntimeError('Verification did not produce a verdict.')
        return verdict

    @staticmethod
    def _build_prompt(subject: object, criteria: tuple[str, ...]) -> str:
        criteria_block = '\n'.join(f'- {c}' for c in criteria)
        return f'SUBJECT:\n{subject}\n\nCRITERIA (all must hold):\n{criteria_block}'

    @staticmethod
    def _log_verdict(verdict: Verdict) -> None:
        logger.info(
            f'Verdict: passed={verdict.passed}, confidence={verdict.confidence:.2f}\nRationale: {verdict.rationale}'
        )
        if verdict.violations:
            logger.info('Violations:\n' + '\n'.join(f'- {v}' for v in verdict.violations))

    @staticmethod
    def _failure_message(verdict: Verdict) -> str:
        message = f'Verification failed (confidence {verdict.confidence:.2f}): {verdict.rationale}'
        if verdict.violations:
            message += '\nViolations:\n' + '\n'.join(f'- {v}' for v in verdict.violations)
        return message
