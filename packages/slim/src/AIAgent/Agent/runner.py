import asyncio
import enum
import functools
import importlib.metadata
import logging
import threading
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Generic, Literal, cast

if TYPE_CHECKING:
    from pydantic_ai._agent_graph import AgentNode
    from pydantic_ai.result import FinalResult
    from pydantic_graph import End

import pydantic_ai
from pydantic_ai import (
    FilePart,
    FinalResultEvent,
    NativeToolCallPart,
    NativeToolReturnPart,
    OutputToolCallEvent,
    OutputToolResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    ThinkingPart,
    ToolCallEvent,
    ToolResultEvent,
    agent,
    capabilities,
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

from AIAgent._literals import literal_values

from .log_format import (
    ContentRenderer,
    LogMessage,
    collapsible,
    contains_media,
    retry_note,
    truncate,
    usage_summary,
)
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
    # every test starts a new event loop, keep its "Using selector" debug message out of the log
    logging.getLogger('asyncio').setLevel(logging.WARNING)


_disable_debug_logging()

# Robot Framework owns the console, keep pydantic-ai's first-run banner out of it
pydantic_ai.BANNER_ENABLED = False

# tool arguments and results longer than this are cut off in the INFO line and logged completely as DEBUG
_TOOL_TEXT_LIMIT = 1000
# length of the tool call repeated in front of its result
_CALL_LABEL_LIMIT = 120


plugins_manager = PluginManager()

if TYPE_CHECKING:
    KnownModelName = enum.Enum('KnownModelName', {'a_model': 'a_model_value'})
    KnownToolName = enum.Enum('KnownToolName', {'a_tool': 'a_tool_value'})
    KnownToolsetName = enum.Enum('KnownToolsetName', {'a_toolset': 'a_toolset_value'})
else:
    KnownModelName = enum.Enum('KnownModelName', {v: v for v in literal_values(PydanticKnownModelName)})
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


class AgentRunner(Generic[output.OutputDataT]):
    """Reusable, provider-agnostic wrapper around a pydantic-ai agent.

    This class owns everything that is *not* Robot Framework specific: building
    the underlying agent, managing the dedicated asyncio event loop in a worker
    thread, streaming and logging run events, and tracking the last run result
    for message-history continuation.

    It is deliberately free of any `robot.api.deco` keyword/library machinery so
    that multiple Robot Framework libraries (e.g. `Agent`, a future `Verify`)
    can share the same core instead of reimplementing it.
    """

    _LOOP_TIMEOUT: float = 30.0  # Timeout for event loop operations in seconds

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
        self._model = model.value if isinstance(model, KnownModelName) else model
        self._output_type = output_type
        self._instructions = instructions
        self._system_prompt = system_prompt
        self._name = name
        self._model_settings = model_settings
        self._retries = retries
        self._output_retries = output_retries
        self._tools = tools
        self._capabilities = capabilities
        self._toolsets = toolsets

        self._last_run_result: agent.AgentRunResult[Any] | None = None

        self._tools_started = False

        self._agent_thread: threading.Thread | None = None
        self._agent_loop: asyncio.AbstractEventLoop | None = None
        self._robot_loop: asyncio.AbstractEventLoop | None = None
        self._renderer = ContentRenderer()

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
            retries=(
                self._retries
                if self._output_retries is None
                else {'tools': self._retries, 'output': self._output_retries}
            ),
            tools=self._build_tools(),
            capabilities=self._capabilities if self._capabilities is not None else [],
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
        End nodes log the run's output and usage, see `_log_result`.

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
            self._log_result(cast('FinalResult[Any]', getattr(node, 'data')), agent_run)
        else:
            # Log unknown node types to detect unexpected graph nodes
            self._log(f'Unknown node type encountered: {type(node).__name__} - {node!r}', 'DEBUG')

    def _log(self, message: str | LogMessage, level: Literal['DEBUG', 'INFO'] = 'INFO') -> None:
        """Hand a message over to Robot Framework's thread, which owns the log."""
        if self._robot_loop is None:
            return
        if isinstance(message, str):
            message = LogMessage(message)
        self._robot_loop.call_soon_threadsafe(functools.partial(logger.write, message.message, level, message.html))

    def _log_result(self, final_result: 'FinalResult[Any]', agent_run: run.AgentRun[output.OutputDataT, Any]) -> None:
        """Log the output and the usage of a finished run.

        A string output that is just the final response's text was already logged when its text part
        completed. Other outputs are logged as JSON, binary outputs such as images as a preview.
        """
        result = final_result.output
        if isinstance(result, str):
            if result != self._last_response_text(agent_run):
                self._log(result)
        else:
            self._log(self._renderer.render(result, prefix='Result: ', indent=2))

        responses = (message for message in agent_run.new_messages() if isinstance(message, messages.ModelResponse))
        self._log(usage_summary(agent_run.usage, responses))

    @staticmethod
    def _last_response_text(agent_run: run.AgentRun[output.OutputDataT, Any]) -> str | None:
        all_messages = agent_run.all_messages()
        last = all_messages[-1] if all_messages else None
        return last.text if isinstance(last, messages.ModelResponse) else None

    def _log_user_prompt(self, node: 'AgentNode[Any, Any]') -> None:
        """Log the user prompt, as DEBUG for plain text and as INFO with previews if it carries files.

        Args:
            node: UserPromptNode with user_prompt attribute
        """
        user_prompt = getattr(node, 'user_prompt', None)
        if user_prompt is None:
            return
        self._log(
            self._renderer.render(user_prompt, prefix='User prompt: '),
            'INFO' if contains_media(user_prompt) else 'DEBUG',
        )

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

        self._log_retry_feedback(node)

        stream_method = getattr(node, 'stream', None)
        if stream_method is None:
            return

        async with stream_method(agent_run.ctx) as request_stream:
            async for event in request_stream:
                match event:
                    case PartEndEvent(part=ThinkingPart() as thinking_part):
                        if thinking_part.content.strip():
                            self._log(collapsible('Thinking', thinking_part.content))
                    case PartEndEvent(part=TextPart() as text_part):
                        self._log(text_part.content)
                    case PartEndEvent(part=FilePart() as file_part):
                        self._log(self._renderer.render(file_part.content, prefix='File from model: '))
                    case PartEndEvent(part=NativeToolCallPart() as tool_call):
                        self._log(
                            f'Native tool call: {tool_call.tool_name}({tool_call.args}) [id={tool_call.tool_call_id}]',
                            'DEBUG',
                        )
                    case PartEndEvent(part=NativeToolReturnPart() as tool_return):
                        prefix = f'Native tool result [{tool_return.tool_call_id}]: '
                        self._log(self._renderer.render(tool_return.content, prefix=prefix), 'DEBUG')
                    case PartEndEvent(part=messages.ToolCallPart()):
                        pass  # logged when the tool is called
                    case PartEndEvent():
                        self._log(f'Request part {event.index} completed: {event.part!r}', 'DEBUG')
                    case FinalResultEvent():
                        # Note: FinalResultEvent fires at the *start* of the final part, before the
                        # remaining text deltas arrive. Do NOT break here, otherwise the stream is
                        # abandoned early and the run result only contains the first chunk.
                        self._log(f'Model producing final result (tool={event.tool_name})', 'DEBUG')
                    case PartStartEvent() | PartDeltaEvent():
                        pass  # the completed part is logged instead
                    case _:
                        # Log unknown event types to detect unexpected model request events
                        self._log(f'Unknown model request event: {type(event).__name__} - {event!r}', 'DEBUG')

    def _log_retry_feedback(self, node: 'AgentNode[Any, Any]') -> None:
        """Log retries that answer no tool call, e.g. a text output that failed validation.

        Args:
            node: ModelRequestNode whose request carries the retry feedback
        """
        request = getattr(node, 'request', None)
        for part in getattr(request, 'parts', ()):
            if isinstance(part, messages.RetryFeedbackPart):
                prefix = f'Retry{retry_note(part.retry)}: '
                self._log(self._renderer.render(part.content, prefix=prefix, limit=_TOOL_TEXT_LIMIT))

    async def _log_tool_calls(
        self, node: 'AgentNode[Any, Any]', agent_run: run.AgentRun[output.OutputDataT, Any]
    ) -> None:
        """Log tool calls and their results as INFO, those of output tools as DEBUG.

        Args:
            node: CallToolsNode with stream() method
            agent_run: Current agent run context
        """
        if self._robot_loop is None:
            return

        stream_method = getattr(node, 'stream', None)
        if stream_method is None:
            return

        # calls by id, so that results of parallel calls of the same tool can be told apart
        calls: dict[str, str] = {}
        async with stream_method(agent_run.ctx) as tool_stream:
            async for event in tool_stream:
                match event:
                    case OutputToolCallEvent():
                        # the output itself is logged at the end of the run
                        self._log(f'Output tool call: {event.part.tool_name}({event.part.args_as_json_str()})', 'DEBUG')
                    case ToolCallEvent():
                        call = f'{event.part.tool_name}({event.part.args_as_json_str()})'
                        calls[event.part.tool_call_id] = call
                        text, truncated = truncate(f'Tool call: {call}', _TOOL_TEXT_LIMIT)
                        self._log(text)
                        if truncated:
                            self._log(f'Tool call: {call} [id={event.part.tool_call_id}]', 'DEBUG')
                    case ToolResultEvent():
                        call = calls.get(event.part.tool_call_id, event.part.tool_name)
                        self._log_tool_result(event, truncate(call, _CALL_LABEL_LIMIT)[0])
                    case _:
                        # Log unknown event types to detect unexpected tool call events
                        self._log(f'Unknown tool call event: {type(event).__name__} - {event!r}', 'DEBUG')

    def _log_tool_result(self, event: ToolResultEvent, call: str) -> None:
        """Log a tool result, cut down and with previews of returned images.

        Retries and failures are logged with their message. Successful results of output tools are only
        logged as DEBUG, the run's output is logged at its end.

        Args:
            event: The tool result event
            call: The tool call the result answers, e.g. `get_weather({"city":"Berlin"})`
        """
        part = event.part
        content: Any = part.content
        if extra := getattr(event, 'content', None):
            # content sent to the model as a user prompt after the result, e.g. images for providers that
            # can't take them in tool results
            content = [content, *extra] if isinstance(extra, (list, tuple)) else [content, extra]

        level: Literal['DEBUG', 'INFO'] = 'INFO'
        match part.outcome:
            case 'success':
                label = f'Tool result: {call}'
                if isinstance(event, OutputToolResultEvent):
                    level = 'DEBUG'
            case 'retried':
                label = f'Tool retry: {call}{retry_note(part.retry)}'
            case outcome:
                label = f'Tool {outcome}: {call}'

        message = self._renderer.render(content, prefix=f'{label}: ', limit=_TOOL_TEXT_LIMIT)
        self._log(message, level)
        if message.truncated:
            prefix = f'{label} (complete, id={part.tool_call_id}): '
            self._log(self._renderer.render(content, prefix=prefix, previews=False), 'DEBUG')

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

    def clear_message_history(self) -> None:
        self._last_run_result = None

    def shutdown(self) -> None:
        """Stop the worker event loop (if running) and reset run state.

        Intended to be called from a Robot Framework library's `end_test` hook to
        prevent cross-test leakage.
        """
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

    async def run(
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
        # resolved here, in Robot Framework's thread, where the output directory is known
        self._renderer = ContentRenderer.for_robot()

        try:
            return await asyncio.wait_for(
                asyncio.wrap_future(
                    asyncio.run_coroutine_threadsafe(
                        self._run_async(
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

    async def _run_async(
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
            # Persist whatever result we have (even on a partial/failed run) so message
            # history can continue. Storing must happen here, but returning must NOT:
            # a `return` inside `finally` would swallow any in-flight exception and let
            # a failed run appear successful.
            if agent_run is not None and agent_run.result is not None:
                self._last_run_result = agent_run.result

        if agent_run is not None and agent_run.result is not None:
            return agent_run.result.output

        return None
