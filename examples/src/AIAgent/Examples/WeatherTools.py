from pydantic_ai import RunContext
from pydantic_ai.toolsets import FunctionToolset

__all__ = ['temperature_celsius', 'temperature_fahrenheit', 'weather_toolset']


def temperature_celsius(city: str) -> float:
    """Get the current temperature in Celsius for a given city."""
    return 21.0


def temperature_fahrenheit(city: str) -> float:
    """Get the current temperature in Fahrenheit for a given city."""
    return 69.8


weather_toolset = FunctionToolset([temperature_celsius, temperature_fahrenheit])


@weather_toolset.tool
def conditions(ctx: RunContext, city: str) -> str:
    if ctx.run_step % 2 == 0:
        return "It's sunny"
    else:
        return "It's raining"
