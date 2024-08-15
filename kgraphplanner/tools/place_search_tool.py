from typing import Callable, Tuple

from langchain_core.tools import tool

from kgraphplanner.tool_manager.abstract_tool import AbstractTool
from kgraphplanner.tool_manager.tool_request import ToolRequest
from kgraphplanner.tool_manager.tool_response import ToolResponse


class PlaceSearchTool(AbstractTool):
    def handle_request(self, request: ToolRequest) -> ToolResponse:
        pass

    def get_sample_text(self) -> str:
        pass

    def get_tool_function(self) -> Callable:

        @tool
        def place_search(location: str) -> Tuple[float, float]:
            """
            Use this to get the latitude and longitude of a location.
            Use format of City Name, State Abbreviation, such as:
            Philadelphia, PA.
            """

            print(f"PlaceSearchTool called with location: {location}")

            return 39.952583, -75.165222

        return place_search

