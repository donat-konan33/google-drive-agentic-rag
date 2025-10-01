# cache and tools ::  https://python.langchain.com/docs/integrations/chat/anthropic/
from langchain_anthropic import convert_to_anthropic_tool
from langchain_core.tools import tool

# For demonstration purposes, we artificially expand the
# tool description.
description = (
    f"Get the weather at a location. By the way, check out this readme: {readme}"
)


@tool(description=description)
def get_weather(location: str) -> str:
    return "It's sunny."


# Enable caching on the tool
weather_tool = convert_to_anthropic_tool(get_weather)
weather_tool["cache_control"] = {"type": "ephemeral"}

llm = ChatAnthropic(model="claude-3-7-sonnet-20250219")
llm_with_tools = llm.bind_tools([weather_tool])
query = "What's the weather in San Francisco?"

response_1 = llm_with_tools.invoke(query)
response_2 = llm_with_tools.invoke(query)

usage_1 = response_1.usage_metadata["input_token_details"]
usage_2 = response_2.usage_metadata["input_token_details"]

print(f"First invocation:\n{usage_1}")
print(f"\nSecond:\n{usage_2}")
