from typing import Callable

from typing_extensions import TypedDict
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


class WeatherReport(TypedDict):
    pass
    # friendly version of WeatherData ?


# TODO: variant that includes place name instead of lat, long and internally
# does the lat, long lookup.  this would have one tool call instead of two.

class WeatherInfoTool(AbstractTool):

    # add cache for recent weather reports?

    def handle_request(self, request: ToolRequest) -> ToolResponse:

        place_label = request.get_parameter("place_label")
        latitude = request.get_parameter("latitude")
        longitude = request.get_parameter("longitude")

        include_previous = request.get_parameter("include_previous")
        use_archive = request.get_parameter("use_archive")
        archive_date = request.get_parameter("archive_date")

        weather_request = WeatherRequest(
            place_label=place_label,
            latitude=latitude,
            longitude=longitude,
            include_previous=include_previous,
            use_archive=use_archive,
            archive_date=archive_date
        )

        tool_endpoint = self.config.get("tool_endpoint")

        tool_config = {
            "tool_endpoint": tool_endpoint
        }

        client = VitalAgentRestResourceClient(tool_config)

        client_tool_response = client.handle_tool_request("weather_tool", weather_request)

        weather_results = client_tool_response.tool_results

        tool_response = ToolResponse()

        tool_response.add_parameter("weather_results", weather_results)

        return tool_response

    def get_sample_text(self) -> str:
        pass

    def get_tool_function(self) -> Callable:

        @tool
        def get_weather(
                place_label: str,
                latitude: float,
                longitude: float,
                include_previous: bool = False,
                use_archive: bool = False,
                archive_date: str = "") -> WeatherData:
            """
            Use this to get weather information for today and the next few days.
            If you set include_previous to True, you can get the last 10 days worth of data.
            If you want the weather for a date older than 10 days ago, then and only then use the archive.

                Attributes:
                    place_label (str): The place label of the weather location.
                    latitude (float)
                    longitude (float)
                    include_previous (bool): include the previous 10 days weather data also
                    use_archive (bool): instead of the current weather, use the archive for historical data
                    archive_date (str): if archive is True, the specific date to use for historical data in format YYYY-MM-DD
            """

            params = {
                'place_label': place_label,
                'latitude': latitude,
                'longitude': longitude,
                'include_previous': include_previous,
                'use_archive': use_archive,
                'archive_date': archive_date
            }

            tool_request = ToolRequest(parameters=params)

            tool_response = self.handle_request(tool_request)

            weather_results: WeatherResponse = tool_response.get_parameter("weather_results")

            weather_data: WeatherData = weather_results.get("weather_data")

            tool_request_guid = weather_data.get("tool_request_guid")

            print(f"get_weather response: tool_request_guid: {tool_request_guid}")

            tool_cache: ToolCache = self.tool_manager.get_tool_cache()
            tool_cache.put_in_cache(tool_request_guid, weather_data)

            return weather_data

        return get_weather
