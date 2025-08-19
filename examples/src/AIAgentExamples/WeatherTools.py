from datetime import datetime

from pydantic_ai import RunContext
from pydantic_ai.toolsets import FunctionToolset

__all__ = ['temperature_celsius', 'temperature_fahrenheit', 'weather_toolset', 'datetime_toolset']


def temperature_celsius(city: str) -> float:
    return 21.0


def temperature_fahrenheit(city: str) -> float:
    return 69.8


weather_toolset = FunctionToolset(tools=[temperature_celsius, temperature_fahrenheit])


@weather_toolset.tool
def conditions(ctx: RunContext, city: str) -> str:
    if ctx.run_step % 2 == 0:
        return "It's sunny"
    else:
        return "It's raining"


datetime_toolset = FunctionToolset()
datetime_toolset.add_function(lambda: datetime.now(), name='now')
