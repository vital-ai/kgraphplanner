from typing import Callable, Tuple
from langchain_core.tools import tool
from vital_agent_kg_utils.vital_agent_rest_resource_client.tools.place_search.place_search_request import \
    PlaceSearchRequest
from vital_agent_kg_utils.vital_agent_rest_resource_client.tools.place_search.place_search_response import \
    PlaceSearchResponse, PlaceSearchData
from vital_agent_kg_utils.vital_agent_rest_resource_client.vital_agent_rest_resource_client import \
    VitalAgentRestResourceClient

from kgraphplanner.tool_manager.abstract_tool import AbstractTool
from kgraphplanner.tool_manager.tool_cache import ToolCache
from kgraphplanner.tool_manager.tool_request import ToolRequest
from kgraphplanner.tool_manager.tool_response import ToolResponse
from typing_extensions import TypedDict


class PlaceSearchTool(AbstractTool):
    def handle_request(self, request: ToolRequest) -> ToolResponse:

        place_search_string = request.get_parameter("place_search_string")

        place_search_request = PlaceSearchRequest(
            place_search_string=place_search_string
        )

        tool_endpoint = self.config.get("tool_endpoint")

        tool_config = {
            "tool_endpoint": tool_endpoint
        }

        client = VitalAgentRestResourceClient(tool_config)

        client_tool_response = client.handle_tool_request("place_search_tool", place_search_request)

        place_search_results = client_tool_response.tool_results

        tool_response = ToolResponse()

        tool_response.add_parameter("place_search_results", place_search_results)

        return tool_response

    def get_sample_text(self) -> str:
        pass

    def get_tool_function(self) -> Callable:

        @tool
        def place_search(place_search_string: str) -> PlaceSearchData:
            """
            Use this to get place data including the latitude and longitude of a location.
            Use format of City Name, State Abbreviation, such as:
            Philadelphia, PA.
            Or, use the full address like:
            123 Main Street, Anytown, NY, USA
            The results will include a list of potential matches.
            You should decide which of these is the one you want.

            Attributes:
                place_search_string (str): The place search string.
            """

            print(f"PlaceSearchTool called with location: {place_search_string}")

            params = {'place_search_string': place_search_string}

            tool_request = ToolRequest(parameters=params)

            tool_response = self.handle_request(tool_request)

            place_search_results: PlaceSearchResponse = tool_response.get_parameter("place_search_results")

            place_search_data: PlaceSearchData = place_search_results.get("place_search_data")

            tool_request_guid = place_search_data.get("tool_request_guid")

            print(f"place_search response: tool_request_guid: {tool_request_guid}")

            tool_cache: ToolCache = self.tool_manager.get_tool_cache()
            tool_cache.put_in_cache(tool_request_guid, place_search_data)

            return place_search_data

        return place_search

