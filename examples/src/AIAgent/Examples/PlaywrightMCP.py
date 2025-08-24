import subprocess
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.shared.message import SessionMessage
from pydantic_ai.mcp import MCPServerStdio

__all__ = ['playwright']


class MCPServerStdioEx(MCPServerStdio):
    @asynccontextmanager
    async def client_streams(
        self,
    ) -> AsyncIterator[
        tuple[
            MemoryObjectReceiveStream[SessionMessage | Exception],
            MemoryObjectSendStream[SessionMessage],
        ]
    ]:
        server = StdioServerParameters(command=self.command, args=list(self.args), env=self.env, cwd=self.cwd)
        async with stdio_client(server=server, errlog=subprocess.STDOUT) as (read_stream, write_stream):
            yield read_stream, write_stream

    def __repr__(self) -> str:
        repr_args = [
            f'command={self.command!r}',
            f'args={self.args!r}',
        ]
        if self.id:
            repr_args.append(f'id={self.id!r}')  # pragma: no cover
        return f'{self.__class__.__name__}({", ".join(repr_args)})'


playwright = MCPServerStdioEx(
    command='npx',
    args=['@playwright/mcp@latest', '--isolated', '--image-responses', 'allow'],
    max_retries=3,
    id='playwright',
)
