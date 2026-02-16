import uuid
from kgraphplanner.vital_agent_rest_resource_client.tools.tool_handler import ToolHandler
from kgraphplanner.vital_agent_rest_resource_client.tools.tool_parameters import ToolParameters
from kgraphplanner.vital_agent_rest_resource_client.tools.tool_results import ToolResults
from kgraphplanner.vital_agent_rest_resource_client.tools.weather.weather_request import WeatherRequest
from kgraphplanner.vital_agent_rest_resource_client.tools.weather.weather_response import WeatherData, \
    WeatherPrediction, WeatherResponse


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

        weather_data = self.parse_weather_response(tool_parameters, response_json)

        weather_response = WeatherResponse(
            weather_data=weather_data
        )

        return weather_response

    def parse_weather_response(self, tool_parameters: WeatherRequest, response_json: dict) -> WeatherData:

        data = response_json

        place_label = tool_parameters.get('place_label', "")

        daily = data['daily']

        daily_predictions = [

            WeatherPrediction(
                date=daily['time'][i],
                weather_code=daily['weather_code'][i],
                weather_code_description=WeatherToolHandler.get_weather_code_id_description(daily['weather_code'][i]),
                temperature_max=daily['temperature_2m_max'][i],
                temperature_min=daily['temperature_2m_min'][i],
                apparent_temperature_max=daily['apparent_temperature_max'][i],
                apparent_temperature_min=daily['apparent_temperature_min'][i],
                sunrise=daily['sunrise'][i],
                sunset=daily['sunset'][i],
                precipitation_sum=daily['precipitation_sum'][i],
                precipitation_hours=daily['precipitation_hours'][i],
                precipitation_probability_max=daily['precipitation_probability_max'][i],
                precipitation_probability_min=daily['precipitation_probability_min'][i],
                precipitation_probability_mean=daily['precipitation_probability_mean'][i],
                daylight_duration=daily['daylight_duration'][i],
                uv_index_max=daily['uv_index_max'][i],
                wind_gusts_10m_max=daily['wind_gusts_10m_max'][i],
            )
            for i in range(len(daily['time']))
        ]

        guid = uuid.uuid4()

        if data.get('current'):

            weather_code=data['current']['weather_code']
            weather_code_description = WeatherToolHandler.get_weather_code_id_description(weather_code)
            weather_data = WeatherData(

                tool_request_guid=str(guid),
                tool_data_class="WeatherData",
                tool_name='weather_tool',
                tool_parameters=tool_parameters,

                place_label=place_label,

                latitude=data['latitude'],
                longitude=data['longitude'],
                timezone=data['timezone'],

                weather_code=weather_code,
                weather_code_description=weather_code_description,

                temperature=data['current']['temperature_2m'],
                humidity=data['current']['relative_humidity_2m'],
                wind_speed=data['current']['wind_speed_10m'],
                apparent_temperature=data['current']['apparent_temperature'],
                is_day=data['current']['is_day'],
                precipitation=data['current']['precipitation'],
                precipitation_probability=data['current']['precipitation_probability'],
                cloud_cover=data['current']['cloud_cover'],
                wind_direction_10m=data['current']['wind_direction_10m'],
                wind_gusts_10m=data['current']['wind_gusts_10m'],

                daily_predictions=daily_predictions
            )
        else:

            weather_data = WeatherData(
                tool_request_guid=str(guid),
                tool_data_class="WeatherData",
                tool_name='weather_tool',
                tool_parameters=tool_parameters,

                place_label=place_label,

                latitude=data['latitude'],
                longitude=data['longitude'],
                timezone=data['timezone'],

                daily_predictions=daily_predictions
            )

        return weather_data

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

