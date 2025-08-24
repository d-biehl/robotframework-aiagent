import inspect
from importlib import metadata
from typing import Any

from pydantic_ai import tools as ai_tools, toolsets as ai_toolsets


class PluginManager:
    def __init__(self) -> None:
        self._tools: dict[str, ai_tools.Tool[Any]] = {}
        self._toolsets: dict[str, ai_toolsets.AbstractToolset[Any]] = {}
        self._load_plugins()

    @property
    def tools(self) -> dict[str, ai_tools.Tool[Any]]:
        return self._tools

    @property
    def toolsets(self) -> dict[str, ai_toolsets.AbstractToolset[Any]]:
        return self._toolsets

    def _load_plugins(self) -> None:
        entry_points = metadata.entry_points(group='AIAgent')
        for entry_point in entry_points:
            plugin = entry_point.load()

            names = [attr for attr in dir(plugin) if not attr.startswith('_')]
            if hasattr(plugin, '__all__'):
                names = [name for name in names if name in plugin.__all__]
            variables = [(name, getattr(plugin, name)) for name in names]
            if not inspect.ismodule(plugin):
                variables = [(n, v) for n, v in variables if not callable(v)]

            for name, variable in variables:
                if isinstance(variable, ai_tools.Tool):
                    self._tools[name] = variable
                elif isinstance(variable, ai_toolsets.AbstractToolset):
                    self._toolsets[name] = variable
                elif callable(variable):
                    self._tools[name] = ai_tools.Tool(variable)
