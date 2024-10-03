from typing import Callable

from typing_extensions import TypedDict
from vital_agent_kg_utils.vital_agent_rest_resource_client.tools.weather.weather_request import WeatherRequest
from vital_agent_kg_utils.vital_agent_rest_resource_client.tools.weather.weather_response import WeatherResponse, \
    WeatherData
from vital_agent_kg_utils.vital_agent_rest_resource_client.vital_agent_rest_resource_client import \
    VitalAgentRestResourceClient

from kgraphplanner.tool_manager.abstract_tool import AbstractTool
from kgraphplanner.tool_manager.tool_request import ToolRequest
from kgraphplanner.tool_manager.tool_response import ToolResponse
from langchain_core.tools import tool


class WeatherReport(TypedDict):
    pass



class WeatherInfoTool(AbstractTool):

    # add cache for recent weather reports?

    def handle_request(self, request: ToolRequest) -> ToolResponse:

        latitude = request.get_parameter("latitude")
        longitude = request.get_parameter("longitude")

        weather_request = WeatherRequest(
            latitude=latitude,
            longitude=longitude
        )

        client = VitalAgentRestResourceClient()

        weather_results = client.handle_tool_request("weather_tool", weather_request)

        tool_response = ToolResponse()

        tool_response.add_parameter("weather_results", weather_results)

        return tool_response

    def get_sample_text(self) -> str:
        pass

    def get_tool_function(self) -> Callable:

        @tool
        def get_weather(latitude: float, longitude: float) -> WeatherData:
            """Use this to get weather information given Latitude, Longitude."""

            params = {'latitude': latitude, 'longitude': longitude}

            tool_request = ToolRequest(parameters=params)

            tool_response = self.handle_request(tool_request)

            weather_results: WeatherResponse = tool_response.get_parameter("weather_results")

            weather_data: WeatherData = weather_results.get("weather_data")

            return weather_data

        return get_weather
