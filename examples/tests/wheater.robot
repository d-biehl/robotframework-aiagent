*** Settings ***
Variables       AIAgent.Examples.WeatherTools
Library         AIAgent.Agent    gpt-5-mini    toolsets=${{[$weather_toolset, "datetime_toolset"]}}    AS    WheatherAgentWithTools
Library         AIAgent.Agent    gpt-5-mini    tools=${{[$temperature_celsius, $temperature_fahrenheit]}}    AS    WheatherAgentWithToolsets


*** Test Cases ***
check tools
    WheatherAgentWithTools.Chat    What tools are available?
    WheatherAgentWithTools.Chat    What is the weather in Berlin?
    WheatherAgentWithTools.Chat    What is the weather in Leipzig?

    WheatherAgentWithTools.Chat    Can you retrieve the weather in Berlin and Leipzig in parallel?

check toolsets
    WheatherAgentWithToolsets.Chat    What tools are available?
    WheatherAgentWithToolsets.Chat    What is the weather in Berlin?
    WheatherAgentWithToolsets.Chat    What is the weather in Leipzig?

    WheatherAgentWithToolsets.Chat    Can you retrieve the weather in Berlin and Leipzig in parallel?
