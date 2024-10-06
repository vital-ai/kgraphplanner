from dotenv import load_dotenv
from vital_agent_kg_utils.vital_agent_rest_resource_client.tools.weather.weather_response import WeatherData, \
    WeatherResponse

from kgraphplanner.tool_manager.tool_manager import ToolManager
from kgraphplanner.tool_manager.tool_request import ToolRequest
from kgraphplanner.tools.weather.weather_info_tool import WeatherInfoTool


def main():
    print("KG Weather Tool Test")

    load_dotenv()

    tool_config = {}
    tool_manager = ToolManager(tool_config)

    weather_tool = WeatherInfoTool({}, tool_manager=tool_manager)

    # test the handle_request as the get_weather function is wrapped with @tool
    # and should be called from langgraph

    latitude = 40.7128
    longitude = -74.0060
    include_previous = False
    use_archive = True
    archive_date = "2020-12-25"

    params = {
        'latitude': latitude,
        'longitude': longitude,
        'include_previous': include_previous,
        'use_archive': use_archive,
        'archive_date': archive_date
    }

    tool_request = ToolRequest(parameters=params)

    tool_response = weather_tool.handle_request(tool_request)

    weather_results: WeatherResponse = tool_response.get_parameter("weather_results")

    weather_data: WeatherData = weather_results.get("weather_data")

    print(weather_data)


if __name__ == "__main__":
    main()
