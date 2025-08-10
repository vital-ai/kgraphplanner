from typing import Callable, Optional, Type
from pydantic import BaseModel, Field
from typing import TypedDict
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


class WeatherParams(BaseModel):
    place_label: str = Field(..., description="The place label of the weather location.")
    latitude: float = Field(..., description="The latitude of the weather location.")
    longitude: float = Field(..., description="The longitude of the weather location.")
    include_previous: bool = Field(False, description="Include the previous 10 days' weather data.")
    use_archive: bool = Field(False, description="Use the archive for historical data.")
    archive_date: str = Field(None, description="The specific date for historical data in 'YYYY-MM-DD' format.")


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

    def get_tool_schema(self) -> Type[BaseModel]:
        return WeatherParams

    def get_tool_function(self) -> Callable:

        @tool("weather-info-tool", args_schema=WeatherParams, return_direct=True)
        def get_weather(
            place_label: str,
            latitude: float,
            longitude: float,
            include_previous: Optional[bool] = False,
            use_archive: Optional[bool] = False,
            archive_date: Optional[str] = None
        ) -> WeatherData:
            """
            Get current and historical weather information for a specified location.
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
