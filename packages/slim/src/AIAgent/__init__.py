import enum
import functools
import importlib.metadata
import logging
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, TypeVar, cast, get_args

from pydantic_ai import Agent, agent, builtin_tools, messages, models, output, run, settings, tools, toolsets, usage
from pydantic_ai.models import KnownModelName as PydanticKnownModelName
from robot.api import logger
from robot.api.deco import keyword, library
from robot.result.model import TestCase as ResultTestCase
from robot.running.model import TestCase as RunningTestCase
from typing_extensions import TypeAliasType

try:
    __version__ = importlib.metadata.version('robotframework-aiagent-slim')
except importlib.metadata.PackageNotFoundError:
    __version__ = '0.0.0'


def _disable_debug_logging() -> None:
    logging.getLogger('pydantic_ai').setLevel(logging.CRITICAL)
    logging.getLogger('pydantic_ai').propagate = True
    logging.getLogger('openai').setLevel(logging.CRITICAL)
    logging.getLogger('openai').propagate = True
    logging.getLogger('anthropic').setLevel(logging.CRITICAL)
    logging.getLogger('anthropic').propagate = True
    logging.getLogger('httpcore').setLevel(logging.CRITICAL)
    logging.getLogger('httpx').propagate = True
    logging.getLogger('httpx').setLevel(logging.CRITICAL)
    logging.getLogger('httpx').propagate = True


_disable_debug_logging()

T = TypeVar('T')


def _get_literal_vals(alias: TypeAliasType) -> frozenset[Any]:
    def val(alias: TypeAliasType) -> Any:
        return alias.__value__

    def args(alias: TypeAliasType) -> tuple[Any, ...]:
        return get_args(val(alias))

    def resolve(alias: TypeAliasType | tuple[T, ...] | T) -> Iterator[T]:
        if isinstance(alias, TypeAliasType):
            for val in resolve(args(alias)):
                yield from resolve(val)
            return
        if isinstance(alias, tuple):
            t_seq = cast(Sequence[T], alias)
            for element in t_seq:
                yield from resolve(element)
            return
        yield alias

    return frozenset(resolve(alias))


# hack to get pydantic-ais known models as enums
if TYPE_CHECKING:
    KnownModelName = enum.Enum('KnownModelName', {'a_model': 'a_model_value'})
else:
    KnownModelName = enum.Enum('KnownModelName', {v: v for v in _get_literal_vals(PydanticKnownModelName)})


class MessageHistoryOptions(enum.Enum):
    """Enumeration of options for selecting message history when running the agent.

    LAST_RUN
        Include the messages produced by the most recent completed agent run
        as the message history for the next agent invocation. If no prior run
        exists, this option results in no message history being provided.
    """

    LAST_RUN = 'last'


def _convert_toolset(value: Any) -> Any:
    return value


@library(
    scope='SUITE',
    version=__version__,
    auto_keywords=True,
    listener='SELF',
    converters={toolsets.AbstractToolset: _convert_toolset},
)
class AIAgent:
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
        tools: list[tools.Tool[Any] | tools.ToolFuncEither[Any, ...]] | None = None,
        builtin_tools: list[builtin_tools.AbstractBuiltinTool] | None = None,
        toolsets: list[Any] | None = None,
    ):
        self._model = model.value if isinstance(model, KnownModelName) else model
        self._output_type = output_type
        self._instructions = instructions
        self._system_prompt = system_prompt
        self._name = name
        self._model_settings = model_settings
        self._retries = retries
        self._output_retries = output_retries
        self._tools = tools
        self._builtin_tools = builtin_tools
        self._toolsets = toolsets

        self._last_run_result: agent.AgentRunResult[Any] | None = None

    @functools.cached_property
    def agent(self) -> Agent[Any, Any]:
        agent = Agent(
            model=self._model,
            output_type=self._output_type,
            instructions=self._instructions,
            system_prompt=self._system_prompt if self._system_prompt is not None else [],
            name=self._name,
            model_settings=cast(settings.ModelSettings, self._model_settings),
            retries=self._retries,
            output_retries=self._output_retries,
            tools=self._tools if self._tools is not None else [],
            builtin_tools=self._builtin_tools if self._builtin_tools is not None else [],
            toolsets=self._toolsets,
        )

        return agent

    def end_test(self, data: RunningTestCase, result: ResultTestCase) -> None:
        self._last_run_result = None

    def _build_message_history(
        self,
        message_history: list[messages.ModelMessage] | MessageHistoryOptions | None,
    ) -> list[messages.ModelMessage] | None:
        if isinstance(message_history, MessageHistoryOptions):
            if message_history == MessageHistoryOptions.LAST_RUN and self._last_run_result is not None:
                return self._last_run_result.all_messages()
            else:
                return None
        return message_history

    @keyword
    def get_message_history(self) -> list[messages.ModelMessage] | None:
        return self._last_run_result.all_messages() if self._last_run_result is not None else None

    @keyword
    async def chat(
        self,
        *user_prompt: messages.UserContent,
        output_type: output.OutputSpec[output.OutputDataT] | None = None,
        message_history: MessageHistoryOptions | list[messages.ModelMessage] | None = MessageHistoryOptions.LAST_RUN,
        model: models.Model | KnownModelName | str | None = None,
        model_settings: settings.ModelSettings | dict[str, Any] | None = None,
        usage_limits: usage.UsageLimits | None = None,
        usage: usage.Usage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[toolsets.AbstractToolset[Any]] | None = None,
    ) -> output.OutputDataT | None:
        agent_run: run.AgentRun[Any, output.OutputDataT] | None = None

        try:
            async with self.agent.iter(
                user_prompt=user_prompt,
                output_type=output_type,
                message_history=self._build_message_history(message_history),
                model=model.value if isinstance(model, KnownModelName) else model,
                model_settings=cast(settings.ModelSettings, model_settings),
                usage_limits=usage_limits,
                usage=usage,
                infer_name=infer_name,
                toolsets=toolsets,
            ) as agent_run:
                async for node in agent_run:
                    logger.debug(f'Agent node: {node}')
        finally:
            if agent_run is not None and agent_run.result is not None:
                self._last_run_result = agent_run.result

                logger.info(str(agent_run.result.output))
                return agent_run.result.output

        return None
