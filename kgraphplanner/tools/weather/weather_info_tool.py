from typing import Callable

from kgraphplanner.tool_manager.abstract_tool import AbstractTool
from kgraphplanner.tool_manager.tool_request import ToolRequest
from kgraphplanner.tool_manager.tool_response import ToolResponse
from langchain_core.tools import tool


class WeatherInfoTool(AbstractTool):

    # add cache for recent weather reports?

    def handle_request(self, request: ToolRequest) -> ToolResponse:
        pass

    def get_sample_text(self) -> str:
        pass

    def get_tool_function(self) -> Callable:

        @tool
        def get_weather(latitude: float, longitude: float) -> str:
            """Use this to get weather information given Latitude, Longitude."""

            return "It's always sunny in philly"

        return get_weather


