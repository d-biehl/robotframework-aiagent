from collections.abc import Sequence
from typing import Any, Generic, cast

from pydantic_ai import (
    capabilities,
    messages,
    models,
    output,
    settings,
    tools,
    toolsets,
    usage,
)
from robot.api.deco import keyword, library
from robot.result.model import TestCase as ResultTestCase
from robot.running.model import TestCase as RunningTestCase

from .runner import (
    AgentRunner,
    HistoryContent,
    HistoryFormat,
    KnownModelName,
    KnownToolName,
    KnownToolsetName,
    MessageHistoryOptions,
    __version__,
)

__all__ = ['Agent']


@library(
    scope='TEST',
    version=__version__,
    auto_keywords=False,
    listener='SELF',
)
class Agent(Generic[output.OutputDataT]):
    """Robot Framework Library wrapper around pydantic-ai agents.

    This library exposes keywords for running model-backed chats and managing
    message history. It is scoped to the current test and resets its last run
    state in `end_test` to prevent cross-test leakage.

    Message history defaults to `LAST_RUN`, meaning consecutive `Chat` calls in
    the same test will continue the prior conversation unless you override it.

    You can configure model settings, tools, and toolsets at construction time
    or override them per call. The return value of `Chat` is the parsed output
    (string or structured type) and is logged to the Robot Framework log.

    The actual run machinery lives in `AgentRunner`; this class is a thin Robot
    Framework binding around it so that sibling libraries can share the core.
    """

    def __init__(
        self,
        model: models.Model | KnownModelName | str | None = None,
        *,
        output_type: output.OutputSpec[output.OutputDataT] = cast(output.OutputSpec[output.OutputDataT], str),
        instructions: str | list[str] | None = None,
        system_prompt: str | list[str] | None = None,
        name: str | None = None,
        model_settings: settings.ModelSettings | dict[str, Any] | None = None,
        retries: int = 1,
        output_retries: int | None = None,
        tools: list[KnownToolName | tools.Tool[Any] | tools.ToolFuncEither]
        | KnownToolName
        | tools.Tool[Any]
        | tools.ToolFuncEither
        | None = None,
        capabilities: Sequence[capabilities.AgentCapability[Any]] | None = None,
        toolsets: list[KnownToolsetName | toolsets.AbstractToolset]
        | KnownToolsetName
        | toolsets.AbstractToolset
        | None = None,
    ):
        self._runner: AgentRunner[output.OutputDataT] = AgentRunner(
            model,
            output_type=output_type,
            instructions=instructions,
            system_prompt=system_prompt,
            name=name,
            model_settings=model_settings,
            retries=retries,
            output_retries=output_retries,
            tools=tools,
            capabilities=capabilities,
            toolsets=toolsets,
        )

    @keyword
    def get_message_history(
        self, content: HistoryContent = HistoryContent.FULL, format: HistoryFormat = HistoryFormat.RAW
    ) -> list[messages.ModelMessage] | str | None:
        return self._runner.get_message_history(content, format)

    @keyword
    def clear_message_history(self) -> None:
        self._runner.clear_message_history()

    def end_test(self, data: RunningTestCase, result: ResultTestCase) -> None:
        self._runner.shutdown()

    @keyword
    async def chat(
        self,
        *user_prompt: messages.UserContent,
        output_type: output.OutputSpec[output.OutputDataT] | None = None,
        message_history: MessageHistoryOptions | list[messages.ModelMessage] | None = MessageHistoryOptions.LAST_RUN,
        model: models.Model | KnownModelName | str | None = None,
        model_settings: settings.ModelSettings | dict[str, Any] | None = None,
        usage_limits: usage.UsageLimits | None = None,
        usage: usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[toolsets.AbstractToolset[Any]] | None = None,
        timeout: float | None = None,
    ) -> output.OutputDataT | None:
        return await self._runner.run(
            *user_prompt,
            output_type=output_type,
            message_history=message_history,
            model=model,
            model_settings=model_settings,
            usage_limits=usage_limits,
            usage=usage,
            infer_name=infer_name,
            toolsets=toolsets,
            timeout=timeout,
        )
