import sys
from pathlib import Path

from fastmcp.client.transports import StdioTransport
from pydantic_ai.mcp import MCPToolset
from robot.libraries.BuiltIn import BuiltIn, RobotNotRunningError

__all__ = ['playwright']


def _output_dir() -> str | None:
    """Return a folder below Robot's output directory for the server's files, or None outside of Robot."""
    try:
        output_dir = BuiltIn().get_variable_value('${OUTPUT_DIR}')
    except RobotNotRunningError:
        return None
    path = Path(output_dir) / 'playwright-mcp'
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


_output = _output_dir()

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
            *(['--output-dir', _output] if _output else []),
        ],
        # files the model saves under an explicit name are resolved against the server's working
        # directory, so run it in Robot's output directory instead of wherever Robot was started
        cwd=_output,
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
