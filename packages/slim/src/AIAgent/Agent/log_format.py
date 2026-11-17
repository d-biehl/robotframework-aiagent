"""Render agent run content for the Robot Framework log.

Binary content never goes into the log as bytes: it is summarized, stored as a file below Robot's
output directory and, for images, shown as a preview.
"""

import hashlib
import html
import mimetypes
import os
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import pydantic_core
from pydantic_ai import messages, usage
from robot.libraries.BuiltIn import BuiltIn, RobotNotRunningError

_PREVIEW_STYLE = 'max-width: 480px; max-height: 360px; border: 1px solid #ccc; margin: 4px 4px 0 0'

Media = messages.BinaryContent | messages.FileUrl


@dataclass(frozen=True)
class LogMessage:
    """A message for `robot.api.logger.write`."""

    message: str
    html: bool = False
    truncated: bool = False


def human_size(size: int) -> str:
    """Format a byte count, e.g. `7.4 MB`."""
    if size < 1000:
        return f'{size} B'
    value = float(size)
    unit = 'B'
    for unit in ('kB', 'MB', 'GB'):
        value /= 1000
        if value < 1000:
            break
    return f'{value:.1f} {unit}'


def summarize(media: Media) -> str:
    """Describe binary content or a file URL in a few words instead of dumping its data."""
    if isinstance(media, messages.BinaryContent):
        return f'[{media.media_type}, {human_size(len(media.data))}]'
    if media.url.startswith('data:'):
        return f'[{_url_media_type(media)}, data URL, {human_size(len(media.url))}]'
    return f'[{_url_media_type(media)}: {media.url}]'


def contains_media(content: Any) -> bool:
    """Whether content carries binary data or file URLs, e.g. an image in a user prompt."""
    if isinstance(content, (messages.BinaryContent, messages.FileUrl)):
        return True
    if isinstance(content, dict):
        return any(contains_media(item) for item in content.values())
    if isinstance(content, (list, tuple)):
        return any(contains_media(item) for item in content)
    return False


def truncate(text: str, limit: int | None) -> tuple[str, bool]:
    """Cut text down to `limit` characters, saying how much was left out."""
    if limit is None or len(text) <= limit:
        return text, False
    return f'{text[:limit]}... [{len(text) - limit:,} more characters]', True


def retry_note(retry: messages.RetryInfo | None) -> str:
    """Say which retry this is, e.g. ` (retry 1 of 3)`."""
    return f' (retry {retry.attempt} of {retry.max_retries})' if retry is not None else ''


def collapsible(summary: str, text: str) -> LogMessage:
    """Render text folded away behind a summary line, e.g. a model's thinking."""
    return LogMessage(
        f'<details><summary>{html.escape(summary)}</summary>'
        f'<div style="white-space: pre-wrap">{html.escape(text)}</div></details>',
        html=True,
    )


def usage_summary(run_usage: usage.RunUsage, responses: Iterable[messages.ModelResponse]) -> str:
    """Summarize requests, tool calls, tokens, the answering models and the cost of an agent run."""
    tokens_in = f'{run_usage.input_tokens:,} input tokens'
    if run_usage.cache_read_tokens:
        tokens_in += f' ({run_usage.cache_read_tokens:,} cached)'
    tokens_out = f'{run_usage.output_tokens:,} output tokens'
    if run_usage.output_reasoning_tokens:
        tokens_out += f' ({run_usage.output_reasoning_tokens:,} reasoning)'

    line = (
        f'Usage: {_count(run_usage.requests, "request")}, {_count(run_usage.tool_calls, "tool call")}, '
        f'{tokens_in}, {tokens_out}'
    )
    models = list(dict.fromkeys(_model_label(response) for response in responses if response.model_name))
    if models:
        line += f'; model: {", ".join(models)}'
    if run_usage.cost is not None:
        line += f'; cost: {_format_cost(run_usage.cost)}'
    return line


class ContentRenderer:
    """Render prompt, tool and model content for the log, storing binary content as files."""

    def __init__(self, files_dir: Path | None = None, log_dir: Path | None = None) -> None:
        self._files_dir = files_dir
        self._log_dir = log_dir

    @classmethod
    def for_robot(cls) -> 'ContentRenderer':
        """Create a renderer storing files in an `aiagent` folder of Robot's output directory.

        Must be called from Robot Framework's thread. Outside of a Robot run binary content is only summarized.
        """
        try:
            builtin = BuiltIn()
            output_dir = builtin.get_variable_value('${OUTPUT DIR}')
            log_file = builtin.get_variable_value('${LOG FILE}')
        except RobotNotRunningError:
            return cls()
        if not output_dir:
            return cls()
        log_dir = Path(log_file).parent if log_file and log_file != 'NONE' else Path(output_dir)
        return cls(Path(output_dir) / 'aiagent', log_dir)

    def render(
        self,
        content: Any,
        *,
        prefix: str = '',
        limit: int | None = None,
        indent: int | None = None,
        previews: bool = True,
    ) -> LogMessage:
        """Render content as a log message: text as it is, other objects as JSON, media summarized.

        Args:
            content: A string, binary content, a file URL, a sequence of those or any JSON-like object.
            prefix: Text put in front of the content.
            limit: Maximum number of characters, longer text is cut off.
            indent: Indentation of content rendered as JSON.
            previews: Whether to show stored images and link stored files.
        """
        media: list[Media] = []
        text, truncated = truncate(prefix + self._text(content, media, indent), limit)
        snippets = [snippet for item in media if (snippet := self._preview(item))] if previews else []
        if not snippets:
            return LogMessage(text, truncated=truncated)
        body = f'<div style="white-space: pre-wrap">{html.escape(text)}</div>'
        return LogMessage(body + ''.join(snippets), html=True, truncated=truncated)

    def _text(self, content: Any, media: list[Media], indent: int | None) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, (messages.BinaryContent, messages.FileUrl)):
            media.append(content)
            return summarize(content)
        if isinstance(content, (list, tuple)) and all(
            isinstance(item, (str, messages.BinaryContent, messages.FileUrl)) for item in content
        ):
            return '\n'.join(self._text(item, media, indent) for item in content)
        return pydantic_core.to_json(_sanitize(content, media), indent=indent, fallback=repr).decode()

    def _preview(self, media: Media) -> str:
        if isinstance(media, messages.FileUrl):
            if media.url.startswith('data:'):
                try:
                    return self._preview(messages.BinaryContent.from_data_uri(media.url))
                except ValueError:
                    return ''
            if isinstance(media, messages.ImageUrl):
                return _image(html.escape(media.url, quote=True))
            return ''

        path = self._store(media)
        if path is None:
            return ''
        href = html.escape(self._link(path), quote=True)
        if media.is_image:
            return _image(href)
        return f'<a href="{href}">{html.escape(summarize(media))}</a> '

    def _store(self, content: messages.BinaryContent) -> Path | None:
        if self._files_dir is None:
            return None
        name = hashlib.sha256(content.data).hexdigest()[:16] + _extension(content.media_type)
        path = self._files_dir / name
        if not path.exists():
            self._files_dir.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content.data)
        return path

    def _link(self, path: Path) -> str:
        return Path(os.path.relpath(path, self._log_dir or path.parent)).as_posix()


def _sanitize(value: Any, media: list[Media]) -> Any:
    if isinstance(value, (messages.BinaryContent, messages.FileUrl)):
        media.append(value)
        return summarize(value)
    if isinstance(value, dict):
        return {key: _sanitize(item, media) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item, media) for item in value]
    return value


def _image(href: str) -> str:
    return f'<a href="{href}"><img src="{href}" style="{_PREVIEW_STYLE}"></a>'


def _url_media_type(url: messages.FileUrl) -> str:
    try:
        return url.media_type
    except ValueError:
        return type(url).__name__


def _extension(media_type: str) -> str:
    return mimetypes.guess_extension(media_type, strict=False) or '.bin'


def _count(number: int, word: str) -> str:
    return f'{number} {word}' + ('' if number == 1 else 's')


def _model_label(response: messages.ModelResponse) -> str:
    if response.provider_name and response.provider_name != response.model_name:
        return f'{response.model_name} ({response.provider_name})'
    return str(response.model_name)


def _format_cost(cost: Decimal) -> str:
    if cost == 0:
        return '$0'
    return f'~${cost:.4f}' if cost >= Decimal('0.0001') else '<$0.0001'
