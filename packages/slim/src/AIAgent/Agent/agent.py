import asyncio
import enum
import functools
import importlib.metadata
import logging
import threading
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, Generic, cast, get_args

if TYPE_CHECKING:
    from pydantic_ai._agent_graph import AgentNode
    from pydantic_ai.result import FinalResult
    from pydantic_graph import End

from pydantic_ai import (
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    FilePart,
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    ThinkingPart,
    agent,
    builtin_tools,
    messages,
    models,
    output,
    run,
    settings,
    tools,
    toolsets,
    usage,
)
from pydantic_ai.models import KnownModelName as PydanticKnownModelName
from robot.api import logger
from robot.api.deco import keyword, library
from robot.api.interfaces import HybridLibrary
from robot.result.model import TestCase as ResultTestCase
from robot.running.model import TestCase as RunningTestCase
from typing_extensions import TypeAliasType

from .plugin_manager import PluginManager

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


def _get_literal_vals(alias: TypeAliasType) -> frozenset[Any]:
    def _alias_value(a: TypeAliasType) -> Any:
        return a.__value__

    def _args(a: TypeAliasType) -> tuple[Any, ...]:
        return get_args(_alias_value(a))

    def _resolve(target: TypeAliasType | tuple[Any, ...] | Any) -> Iterator[Any]:
        if isinstance(target, TypeAliasType):
            for item in _resolve(_args(target)):
                yield from _resolve(item)
            return
        if isinstance(target, tuple):
            for element in target:
                yield from _resolve(element)  # pyright: ignore[reportUnknownArgumentType]
            return
        yield target

    return frozenset(_resolve(alias))


plugins_manager = PluginManager()

if TYPE_CHECKING:
    KnownModelName = enum.Enum('KnownModelName', {'a_model': 'a_model_value'})
    KnownToolName = enum.Enum('KnownToolName', {'a_tool': 'a_tool_value'})
    KnownToolsetName = enum.Enum('KnownToolsetName', {'a_toolset': 'a_toolset_value'})
else:
    KnownModelName = enum.Enum('KnownModelName', {v: v for v in _get_literal_vals(PydanticKnownModelName)})
    KnownToolName = enum.Enum('KnownToolName', {v: v for v in plugins_manager.tools.keys()})
    KnownToolsetName = enum.Enum('KnownToolsetName', {v: v for v in plugins_manager.toolsets.keys()})


class MessageHistoryOptions(enum.Enum):
    """Enumeration of options for selecting message history when running the agent.

    LAST_RUN
        Include the messages produced by the most recent completed agent run
        as the message history for the next agent invocation. If no prior run
        exists, this option results in no message history being provided.
    """

    LAST_RUN = 'last'


class HistoryFormat(enum.Enum):
    """Enumeration of formats for result messages."""

    JSON = 'json'
    RAW = 'raw'


class HistoryContent(enum.Enum):
    """Enumeration of formats for result messages."""

    NEWEST = 'newest'
    FULL = 'all'


@library(
    scope='TEST',
    version=__version__,
    auto_keywords=True,
    listener='SELF',
)
class Agent(HybridLibrary, Generic[output.OutputDataT]):
    """Robot Framework Library wrapper around pydantic-ai agents.

    This library exposes keywords for running model-backed chats and managing
    message history. It is scoped to the current test and resets its last run
    state in `end_test` to prevent cross-test leakage.

    Message history defaults to `LAST_RUN`, meaning consecutive `Chat` calls in
    the same test will continue the prior conversation unless you override it.

    You can configure model settings, tools, and toolsets at construction time
    or override them per call. The return value of `Chat` is the parsed output
    (string or structured type) and is logged to the Robot Framework log.
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
        builtin_tools: list[builtin_tools.AbstractBuiltinTool] | None = None,
        toolsets: list[KnownToolsetName | toolsets.AbstractToolset]
        | KnownToolsetName
        | toolsets.AbstractToolset
        | None = None,
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

        self._tools_started = False

        self._agent_thread: threading.Thread | None = None
        self._agent_loop: asyncio.AbstractEventLoop | None = None
        self._robot_loop: asyncio.AbstractEventLoop | None = None

        self._agent_loop_stop: threading.Event | None = None
        self._agent_loop_running: threading.Event | None = None

    def _build_tools(self) -> list[tools.Tool[Any]]:
        if self._tools is None:
            return []

        if isinstance(self._tools, KnownToolName):
            tool = plugins_manager.tools.get(self._tools.name)
            if tool is not None:
                return [tool]
            raise ValueError(f'Unknown tool name: {self._tools.name}')

        if isinstance(self._tools, tools.Tool):
            return [self._tools]

        if callable(self._tools):
            return [tools.Tool(self._tools)]

        if not isinstance(self._tools, list):
            raise TypeError(f'Unsupported tools type: {type(self._tools)}')

        result = []
        for entry in self._tools:
            if isinstance(entry, KnownToolName):
                tool = plugins_manager.tools.get(entry.name)
                if tool is not None:
                    result.append(tool)
            elif isinstance(entry, tools.Tool):
                result.append(entry)
            elif callable(entry):
                result.append(tools.Tool(entry))
            else:
                raise TypeError(f'Unsupported tool type: {entry} ({type(entry)})')
        return result

    def _build_toolsets(self) -> list[toolsets.AbstractToolset[Any]]:
        if self._toolsets is None:
            return []

        if isinstance(self._toolsets, KnownToolsetName):
            toolset = plugins_manager.toolsets.get(self._toolsets.name)
            if toolset is not None:
                return [toolset]
            raise ValueError(f'Unknown toolset name: {self._toolsets.name}')

        if isinstance(self._toolsets, toolsets.AbstractToolset):
            return [self._toolsets]

        result = []
        for entry in self._toolsets:
            if isinstance(entry, KnownToolsetName):
                toolset = plugins_manager.toolsets.get(entry.name)
                if toolset is not None:
                    result.append(toolset)
            elif isinstance(entry, toolsets.AbstractToolset):
                result.append(entry)
            else:
                raise TypeError(f'Unsupported toolset type: {entry} ({type(entry)})')
        return result

    @functools.cached_property
    def _agent(self) -> agent.Agent[Any, Any]:
        result = agent.Agent(
            model=self._model,
            output_type=self._output_type,
            instructions=self._instructions,
            system_prompt=self._system_prompt if self._system_prompt is not None else [],
            name=self._name,
            model_settings=cast(settings.ModelSettings, self._model_settings),
            retries=self._retries,
            output_retries=self._output_retries,
            tools=self._build_tools(),
            builtin_tools=self._builtin_tools if self._builtin_tools is not None else [],
            toolsets=self._build_toolsets(),
        )

        return result

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

    async def _log_node(
        self,
        node: 'AgentNode[Any, output.OutputDataT] | End[FinalResult[output.OutputDataT]]',
        agent_run: run.AgentRun[output.OutputDataT, Any],
    ) -> None:
        """Log agent nodes with different log levels based on node type.

        Uses runtime type guards to determine the specific node type and route to appropriate loggers.
        End nodes with string output are logged as INFO, representing the final agent result.

        Args:
            node: Agent graph node (UserPromptNode, ModelRequestNode, CallToolsNode, or End)
            agent_run: Current agent run context
        """
        if self._robot_loop is None:
            raise RuntimeError('Robot event loop not available')

        if agent.Agent.is_user_prompt_node(node):
            self._log_user_prompt(node)
        elif agent.Agent.is_model_request_node(node):
            await self._log_model_request(node, agent_run)
        elif agent.Agent.is_call_tools_node(node):
            await self._log_tool_calls(node, agent_run)
        elif hasattr(node, 'data') and hasattr(getattr(node, 'data', None), 'output'):
            # End node with FinalResult - log output if it's a string
            final_result = cast('FinalResult[Any]', getattr(node, 'data'))
            if isinstance(final_result.output, str):
                self._robot_loop.call_soon_threadsafe(logger.info, final_result.output)
        else:
            # Log unknown node types to detect unexpected graph nodes
            self._robot_loop.call_soon_threadsafe(
                logger.debug, f'Unknown node type encountered: {type(node).__name__} - {node!r}'
            )

    def _log_user_prompt(self, node: 'AgentNode[Any, Any]') -> None:
        """Log user prompt as DEBUG.

        Args:
            node: UserPromptNode with user_prompt attribute
        """
        if self._robot_loop is not None:
            # Access user_prompt attribute dynamically
            user_prompt_text = getattr(node, 'user_prompt', str(node))
            self._robot_loop.call_soon_threadsafe(logger.debug, f'User prompt: {user_prompt_text!r}')

    async def _log_model_request(
        self, node: 'AgentNode[Any, Any]', agent_run: run.AgentRun[output.OutputDataT, Any]
    ) -> None:
        """Stream and log model request events.

        Args:
            node: ModelRequestNode with stream() method
            agent_run: Current agent run context
        """
        if self._robot_loop is None:
            return

        stream_method = getattr(node, 'stream', None)
        if stream_method is None:
            return

        async with stream_method(agent_run.ctx) as request_stream:
            async for event in request_stream:
                match event:
                    case PartStartEvent():
                        self._robot_loop.call_soon_threadsafe(
                            logger.debug, f'Request part {event.index} started: {event.part!r}'
                        )
                    case PartEndEvent(part=ThinkingPart() as thinking_part):
                        # Log ThinkingPart content as INFO with thinking marker (reasoning/thinking output)
                        self._robot_loop.call_soon_threadsafe(logger.info, f'[THINKING] {thinking_part.content}')
                        self._robot_loop.call_soon_threadsafe(
                            logger.debug, f'Request part {event.index} completed: {event.part!r}'
                        )
                    case PartEndEvent(part=TextPart() as text_part):
                        # Log TextPart content as INFO (final text response)
                        self._robot_loop.call_soon_threadsafe(logger.info, text_part.content)
                    case PartEndEvent(part=FilePart() as file_part):
                        # Log FilePart (multimodal content like images/documents) as INFO
                        content_type = getattr(file_part.content, 'media_type', 'unknown')
                        content_size = len(getattr(file_part.content, 'data', b''))
                        self._robot_loop.call_soon_threadsafe(
                            logger.info,
                            f'[FILE] Received {content_type} file ({content_size} bytes)'
                            + (f' [id={file_part.id}]' if file_part.id else ''),
                        )
                    case PartEndEvent(part=BuiltinToolCallPart() as tool_call):
                        # Log builtin tool calls as DEBUG
                        self._robot_loop.call_soon_threadsafe(
                            logger.debug,
                            f'Builtin tool call: {tool_call.tool_name}({tool_call.args}) [id={tool_call.tool_call_id}]',
                        )
                    case PartEndEvent(part=BuiltinToolReturnPart() as tool_return):
                        # Log builtin tool returns as DEBUG
                        self._robot_loop.call_soon_threadsafe(
                            logger.debug,
                            f'Builtin tool result [{tool_return.tool_call_id}]: {tool_return.content}',
                        )
                    case PartEndEvent():
                        self._robot_loop.call_soon_threadsafe(
                            logger.debug, f'Request part {event.index} completed: {event.part!r}'
                        )
                    case FinalResultEvent():
                        self._robot_loop.call_soon_threadsafe(
                            logger.debug, f'Model producing final result (tool={event.tool_name})'
                        )
                        break
                    case PartDeltaEvent():
                        pass  # Intentionally ignore delta events (streaming chunks)
                    case _:
                        # Log unknown event types to detect unexpected model request events
                        self._robot_loop.call_soon_threadsafe(
                            logger.debug, f'Unknown model request event: {type(event).__name__} - {event!r}'
                        )

    async def _log_tool_calls(
        self, node: 'AgentNode[Any, Any]', agent_run: run.AgentRun[output.OutputDataT, Any]
    ) -> None:
        """Log tool calls and results as DEBUG.

        Args:
            node: CallToolsNode with stream() method
            agent_run: Current agent run context
        """
        if self._robot_loop is None:
            return

        stream_method = getattr(node, 'stream', None)
        if stream_method is None:
            return

        async with stream_method(agent_run.ctx) as tool_stream:
            async for event in tool_stream:
                match event:
                    case FunctionToolCallEvent():
                        self._robot_loop.call_soon_threadsafe(
                            logger.debug,
                            f'Tool call: {event.part.tool_name}({event.part.args}) [id={event.part.tool_call_id}]',
                        )
                    case FunctionToolResultEvent():
                        self._robot_loop.call_soon_threadsafe(
                            logger.debug,
                            f'Tool result [{event.tool_call_id}]: {event.result.content}',
                        )
                    case _:
                        # Log unknown event types to detect unexpected tool call events
                        self._robot_loop.call_soon_threadsafe(
                            logger.debug, f'Unknown tool call event: {type(event).__name__} - {event!r}'
                        )

    @keyword
    def get_message_history(
        self, content: HistoryContent = HistoryContent.FULL, format: HistoryFormat = HistoryFormat.RAW
    ) -> list[messages.ModelMessage] | str | None:
        if self._last_run_result is None:
            return None

        match (content, format):
            case (HistoryContent.NEWEST, HistoryFormat.RAW):
                return self._last_run_result.new_messages()
            case (HistoryContent.NEWEST, HistoryFormat.JSON):
                return self._last_run_result.new_messages_json().decode('utf-8')
            case (HistoryContent.FULL, HistoryFormat.RAW):
                return self._last_run_result.all_messages()
            case (HistoryContent.FULL, HistoryFormat.JSON):
                return self._last_run_result.all_messages_json().decode('utf-8')

        raise ValueError(f'Unsupported content/format combination: {content}/{format}')

    @keyword
    def clear_message_history(self) -> None:
        self._last_run_result = None

    def end_test(self, data: RunningTestCase, result: ResultTestCase) -> None:
        if self._agent_loop_stop is not None:
            self._agent_loop_stop.set()

        if self._agent_loop_running is not None:
            self._agent_loop_running.wait(30)

        self.clear_message_history()

    def _ensure_agent_loop(self) -> asyncio.AbstractEventLoop:
        if self._agent_loop is not None:
            return self._agent_loop

        # Wait for any previous loop to shut down completely
        if self._agent_loop_running is not None:
            self._agent_loop_running.wait(self._LOOP_TIMEOUT)

        return self._start_agent_loop()

    _LOOP_TIMEOUT: float = 30.0  # Timeout for event loop operations in seconds

    def _start_agent_loop(self) -> asyncio.AbstractEventLoop:
        """Start a new agent event loop in a separate thread."""
        initialized_event = threading.Event()
        exception: Exception | None = None

        def agent_main() -> None:
            nonlocal exception
            try:
                asyncio.run(self._run_agent_loop(initialized_event))
            except Exception as e:
                exception = e
                initialized_event.set()
                self._cleanup_on_error()

        self._agent_thread = threading.Thread(target=agent_main, daemon=True)
        self._agent_thread.start()

        # Wait for initialization with timeout
        if not initialized_event.wait(self._LOOP_TIMEOUT):
            self._cleanup_on_error()
            raise RuntimeError('Agent event loop initialization timed out')

        if exception is not None:
            self._cleanup_on_error()
            raise RuntimeError(f'Failed to start agent event loop: {exception}') from exception

        if self._agent_loop is None:
            self._cleanup_on_error()
            raise RuntimeError('Agent event loop was not properly initialized')

        return self._agent_loop

    async def _run_agent_loop(self, initialized_event: threading.Event) -> None:
        """Run the agent event loop until stop is requested."""
        try:
            self._agent_loop_running = threading.Event()
            self._agent_loop = asyncio.get_event_loop()
            self._agent_loop_stop = threading.Event()

            async with self._agent:
                initialized_event.set()

                # Keep the loop alive until stop is requested
                while not self._agent_loop_stop.is_set():
                    await asyncio.sleep(0.1)  # Slightly longer sleep to reduce CPU usage
        finally:
            # Clean up resources
            self._agent_loop = None
            self._agent_thread = None
            if self._agent_loop_running is not None:
                self._agent_loop_running.set()

    def _cleanup_on_error(self) -> None:
        """Clean up resources when an error occurs during loop initialization."""
        self._agent_loop = None
        self._agent_thread = None
        if self._agent_loop_running is not None:
            self._agent_loop_running.set()

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
        self._robot_loop = asyncio.get_event_loop()

        try:
            return await asyncio.wait_for(
                asyncio.wrap_future(
                    asyncio.run_coroutine_threadsafe(
                        self._chat_async(
                            *user_prompt,
                            output_type=output_type,
                            message_history=message_history,
                            model=model,
                            model_settings=model_settings,
                            usage_limits=usage_limits,
                            usage=usage,
                            infer_name=infer_name,
                            toolsets=toolsets,
                        ),
                        self._ensure_agent_loop(),
                    ),
                ),
                timeout=timeout,
            )
        finally:
            self._robot_loop = None

    async def _chat_async(
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
    ) -> output.OutputDataT | None:
        agent_run: run.AgentRun[Any, output.OutputDataT] | None = None

        try:
            async with self._agent.iter(
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
                    await self._log_node(node, agent_run)
        finally:
            if agent_run is not None and agent_run.result is not None:
                self._last_run_result = agent_run.result
                return agent_run.result.output

        return None

    @functools.cached_property
    def _static_keywords(self) -> Sequence[str]:
        return [name for name in dir(self) if self._is_static_keyword(name)]

    def _is_static_keyword(self, name: str) -> bool:
        """Check if the given name is a keyword in this library."""
        if name.startswith('_'):
            return False

        try:
            o = object.__getattribute__(self, name)  # bypass __getattr__
        except AttributeError:
            return False

        return callable(o) and hasattr(o, 'robot_name') and not hasattr(o, 'robot_not_keyword')

    @functools.cached_property
    def _all_keywords(self) -> Sequence[str]:
        return self._static_keywords

    def get_keyword_names(self) -> Sequence[str]:
        return self._all_keywords

    @functools.cached_property
    def _modes(self) -> dict[str, Any]:
        return {'ask': None}

    def __getattr__(self, name: str) -> Any:
        if name in self._modes:
            return self._modes[name]

        raise AttributeError(name)
