import uuid
from kgraphplanner.vital_agent_rest_resource_client.tools.tool_handler import ToolHandler
from kgraphplanner.vital_agent_rest_resource_client.tools.tool_parameters import ToolParameters
from kgraphplanner.vital_agent_rest_resource_client.tools.tool_results import ToolResults
from .models import WeatherInput, WeatherOutput, WeatherData


class WeatherToolHandler(ToolHandler):

    weather_code_mapping = {
        0: "wi-day-sunny",  # Clear sky
        1: "wi-day-sunny-overcast",  # Mainly clear
        2: "wi-day-cloudy",  # Partly cloudy
        3: "wi-day-cloudy",  # Overcast
        45: "wi-fog",  # Fog
        48: "wi-fog",  # Depositing rime fog
        51: "wi-day-showers",  # Drizzle: Light
        53: "wi-day-showers",  # Drizzle: Moderate
        55: "wi-day-storm-showers",  # Drizzle: Dense
        61: "wi-day-rain",  # Rain: Slight
        63: "wi-day-rain",  # Rain: Moderate
        65: "wi-day-rain",  # Rain: Heavy
        66: "wi-day-sleet",  # Freezing rain: Light
        67: "wi-day-sleet",  # Freezing rain: Heavy
        71: "wi-day-snow",  # Snow fall: Slight
        73: "wi-day-snow",  # Snow fall: Moderate
        75: "wi-day-snow",  # Snow fall: Heavy
        77: "wi-snowflake-cold",  # Snow grains
        80: "wi-day-showers",  # Rain showers: Slight
        81: "wi-day-showers",  # Rain showers: Moderate
        82: "wi-day-showers",  # Rain showers: Violent
        85: "wi-day-snow",  # Snow showers: Slight
        86: "wi-day-snow",  # Snow showers: Heavy
        95: "wi-thunderstorm",  # Thunderstorm: Slight or moderate
        96: "wi-thunderstorm",  # Thunderstorm with slight hail
        99: "wi-thunderstorm",  # Thunderstorm with heavy hail
    }

    weather_code_description_mapping = {
        0: "Sunny",
        1: "Sunny and Overcast",
        2: "Partly cloudy",
        3: "Cloudy",
        45: "Fog",
        48: "Depositing rime fog",
        51: "Light Rain",
        53: "Moderate Rain",
        55: "Storm Showers",
        61: "Slight Rain",
        63: "Moderate Rain",
        65: "Heavy Rain",
        66: "Sleet and Freezing Rain",
        67: "Heavy Sleet and Freezing Rain",
        71: "Light Snow",
        73: "Moderate Snow",
        75: "Heavy Snow",
        77: "Snowflakes",
        80: "Slight Rain Showers",
        81: "Moderate Rain Showers",  # Rain showers: Moderate
        82: "Heavy Rain Showers",  # Rain showers: Violent
        85: "Slight Snow",  # Snow showers: Slight
        86: "Heavy Snow",  # Snow showers: Heavy
        95: "Thunderstorm",  # Thunderstorm: Slight or moderate
        96: "Thunderstorm and Hail",  # Thunderstorm with slight hail
        99: "Thunderstorm and Heavy Hail",  # Thunderstorm with heavy hail

    }

    def handle_response(self, tool_parameters: ToolParameters, response_json: dict) -> ToolResults:

        weather_output = self.parse_weather_response(tool_parameters, response_json)
        return weather_output

    def parse_weather_response(self, tool_parameters: ToolParameters, response_json: dict) -> WeatherOutput:

        # Extract weather data from tool_output structure
        weather_data = response_json['tool_output']['weather_data']

        # Create WeatherData object from the response
        weather_data_obj = WeatherData(
            latitude=weather_data['latitude'],
            longitude=weather_data['longitude'],
            timezone=weather_data['timezone'],
            current=weather_data.get('current'),
            daily=weather_data.get('daily'),
            hourly=weather_data.get('hourly')
        )

        # Return WeatherOutput with the required structure
        return WeatherOutput(
            tool="weather_tool",
            weather_data=weather_data_obj
        )

    @staticmethod
    def get_weather_code_id(weather_code: int) -> str:
        """
        Returns the identifier corresponding to the weather code.

        Args:
            weather_code (int): The weather condition code.

        Returns:
            str: The identifier corresponding to the weather code.
        """

        return WeatherToolHandler.weather_code_mapping.get(weather_code, "wi-na")

    @staticmethod
    def get_weather_code_id_description(weather_code: int) -> str:
        """
        Returns the description corresponding to the weather code.

        Args:
            weather_code (int): The weather condition code.

        Returns:
            str: The weather description corresponding to the weather code.
        """

        return WeatherToolHandler.weather_code_description_mapping.get(weather_code, "Unknown Weather Code")

