import sys

from fastmcp.client.transports import StdioTransport
from pydantic_ai.mcp import MCPToolset

__all__ = ['playwright']


playwright = MCPToolset(
    StdioTransport(
        command='npx',
        args=[
            '@playwright/mcp@latest',
            '--isolated',
            '--image-responses',
            'allow',
            # '--browser',
            # 'firefox',
        ],
        # Robot Framework swaps sys.stderr for a capture stream without a file descriptor
        # while keywords run, so hand the server process the real stderr instead
        log_file=sys.__stderr__,
        # AgentRunner uses a fresh event loop per test; a kept-alive session would still be
        # bound to the previous test's closed loop
        keep_alive=False,
    ),
    max_retries=3,
    id='playwright',
)
