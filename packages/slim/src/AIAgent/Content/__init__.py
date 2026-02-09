from os import PathLike
from pathlib import Path

from pydantic_ai import (
    AudioUrl,
    BinaryContent,
    DocumentUrl,
    ImageUrl,
    VideoUrl,
)

__all__ = [
    'load_binary_content',
    'ImageUrl',
    'AudioUrl',
    'DocumentUrl',
    'VideoUrl',
    'BinaryContent',
]


def load_binary_content(path: PathLike[str], mime_type: str | None = None) -> BinaryContent:
    """Load binary content from a file path and infer MIME type when needed."""
    file_path = Path(path)
    resolved_mime_type = mime_type or _guess_mime_type(file_path)
    return BinaryContent(data=file_path.read_bytes(), media_type=resolved_mime_type) # pyright: ignore[reportCallIssue]


def _guess_mime_type(file_path: Path) -> str:
    """Return a MIME type for common AI-model supported formats."""
    suffix = file_path.suffix.lower()
    explicit_map = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.webp': 'image/webp',
        '.gif': 'image/gif',
        '.wav': 'audio/wav',
        '.mp3': 'audio/mpeg',
        '.mpeg': 'audio/mpeg',
        '.ogg': 'audio/ogg',
        '.flac': 'audio/flac',
        '.mp4': 'video/mp4',
        '.webm': 'video/webm',
        '.mov': 'video/quicktime',
        '.pdf': 'application/pdf',
        '.txt': 'text/plain',
        '.md': 'text/markdown',
        '.html': 'text/html',
        '.htm': 'text/html',
        '.csv': 'text/csv',
        '.json': 'application/json',
    }
    if suffix in explicit_map:
        return explicit_map[suffix]

    from mimetypes import guess_type

    guessed, _ = guess_type(str(file_path))
    return guessed or 'application/octet-stream'
