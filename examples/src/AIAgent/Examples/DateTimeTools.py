from datetime import datetime

from pydantic_ai.toolsets import FunctionToolset

__all__ = ['datetime_toolset']


datetime_toolset = FunctionToolset()
datetime_toolset.add_function(lambda: datetime.now(), name='now')
