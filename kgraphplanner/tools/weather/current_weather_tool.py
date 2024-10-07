from typing import Callable, List, Union

from typing_extensions import TypedDict
from vital_agent_kg_utils.vital_agent_rest_resource_client.tools.place_search.place_search_request import \
    PlaceSearchRequest
from vital_agent_kg_utils.vital_agent_rest_resource_client.tools.place_search.place_search_response import PlaceDetails, \
    PlaceSearchData
from vital_agent_kg_utils.vital_agent_rest_resource_client.tools.weather.weather_request import WeatherRequest
from vital_agent_kg_utils.vital_agent_rest_resource_client.tools.weather.weather_response import WeatherResponse, \
    WeatherData
from vital_agent_kg_utils.vital_agent_rest_resource_client.vital_agent_rest_resource_client import \
    VitalAgentRestResourceClient

from kgraphplanner.tool_manager.tool_cache import ToolCache
from kgraphplanner.tool_manager.abstract_tool import AbstractTool
from kgraphplanner.tool_manager.tool_request import ToolRequest
from kgraphplanner.tool_manager.tool_response import ToolResponse
from langchain_core.tools import tool


class CurrentWeatherTool(AbstractTool):

    # add cache for recent places and weather reports?

    def handle_request(self, request: ToolRequest) -> ToolResponse:

        tool_endpoint = self.config.get("tool_endpoint")

        tool_config = {
            "tool_endpoint": tool_endpoint
        }

        client = VitalAgentRestResourceClient(tool_config)

        # call place and then weather

        place_name = request.get_parameter("place_name")

        place_search_request = PlaceSearchRequest(
            place_search_string=place_name
        )

        client_tool_response = client.handle_tool_request("place_search_tool", place_search_request)

        place_search_results = client_tool_response.tool_results

        place_search_data: Union[PlaceSearchData|None] = place_search_results.get('place_search_data', None)

        place_details_list: List[PlaceDetails] = place_search_data.get('place_details_list', [])

        # just pick the first one
        place_details = place_details_list[0] if place_details_list else None

        if place_details:

            latitude = place_details.get('latitude')
            longitude = place_details.get('longitude')

            weather_request = WeatherRequest(
                place_label=place_name,
                latitude=latitude,
                longitude=longitude,
                include_previous=False,
                use_archive=False,
                archive_date=""
            )

            client_tool_response = client.handle_tool_request("weather_tool", weather_request)

            weather_results = client_tool_response.tool_results

            tool_response = ToolResponse()

            tool_response.add_parameter("weather_results", weather_results)

            return tool_response

        # empty
        tool_response = ToolResponse()

        return tool_response

    def get_sample_text(self) -> str:
        pass

    def get_tool_function(self) -> Callable:

        @tool
        def get_current_weather(place_name: str) -> WeatherData:
            """
            Use this to get weather information for today and the next few days given the name of a place.

                Attributes:
                    place_name (str): The place name of the weather location.
            """

            params = {
                'place_name': place_name,
            }

            tool_request = ToolRequest(parameters=params)

            tool_response = self.handle_request(tool_request)

            weather_results: WeatherResponse = tool_response.get_parameter("weather_results")

            weather_data: WeatherData = weather_results.get("weather_data")

            tool_request_guid = weather_data.get("tool_request_guid")

            print(f"get_current_weather response: tool_request_guid: {tool_request_guid}")

            tool_cache: ToolCache = self.tool_manager.get_tool_cache()
            tool_cache.put_in_cache(tool_request_guid, weather_data)

            return weather_data

        return get_current_weather
