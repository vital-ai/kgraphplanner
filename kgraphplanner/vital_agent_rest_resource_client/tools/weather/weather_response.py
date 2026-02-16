from typing import List, Optional, Annotated
from typing_extensions import TypedDict
from kgraphplanner.vital_agent_rest_resource_client.tools.tool_data import ToolData
from kgraphplanner.vital_agent_rest_resource_client.tools.tool_results import ToolResults


class WeatherPrediction(TypedDict):
    """Represents a daily weather prediction."""

    date: Annotated[str, ..., "The date of the prediction in ISO8601 format."]
    weather_code: Annotated[int, ..., "The weather condition code."]
    weather_code_description: Annotated[Optional[str], None, "The weather condition description."]
    temperature_max: Annotated[float, ..., "The maximum temperature for the day."]
    temperature_min: Annotated[float, ..., "The minimum temperature for the day."]
    apparent_temperature_max: Annotated[float, ..., "The maximum apparent temperature for the day."]
    apparent_temperature_min: Annotated[float, ..., "The minimum apparent temperature for the day."]
    sunrise: Annotated[str, ..., "The time of sunrise in ISO8601 format."]
    sunset: Annotated[str, ..., "The time of sunset in ISO8601 format."]
    precipitation_sum: Annotated[float, ..., "The precipitation sum for the day."]
    precipitation_hours: Annotated[float, ..., "The precipitation hours for the day."]
    precipitation_probability_max: Annotated[float, ..., "The maximum precipitation probability for the day."]
    precipitation_probability_min: Annotated[float, ..., "The minimum precipitation probability for the day."]
    precipitation_probability_mean: Annotated[float, ..., "The mean precipitation probability for the day."]
    daylight_duration: Annotated[float, ..., "The daylight duration for the day."]
    uv_index_max: Annotated[float, ..., "The maximum UV index for the day."]
    wind_gusts_10m_max: Annotated[float, ..., "The maximum wind gusts for the day."]


class WeatherData(ToolData):
    """Represents the weather data including current conditions and daily predictions."""

    place_label: Annotated[str, ..., "The place label of the weather prediction."]
    latitude: Annotated[float, ..., "The latitude of the location."]
    longitude: Annotated[float, ..., "The longitude of the location."]
    timezone: Annotated[str, ..., "The timezone of the location."]

    temperature: Annotated[Optional[float], None, "The current temperature at the location."]
    humidity: Annotated[Optional[int], None, "The current relative humidity at the location."]
    wind_speed: Annotated[Optional[float], None, "The current wind speed at the location."]

    weather_code: Annotated[Optional[int], None, "The weather condition code."]
    weather_code_description: Annotated[Optional[str], None, "The weather condition description."]
    apparent_temperature: Annotated[Optional[float], None, "The apparent temperature at the location."]
    is_day: Annotated[Optional[int], None, "Indicates if it is daytime (1 for daytime, 0 for nighttime)."]
    precipitation: Annotated[Optional[float], None, "The current precipitation at the location."]
    precipitation_probability: Annotated[
        Optional[float], None, "The current precipitation probability at the location."]
    cloud_cover: Annotated[Optional[float], None, "The current cloud cover at the location."]
    wind_direction_10m: Annotated[Optional[float], None, "The wind direction at 10 meters above ground level."]
    wind_gusts_10m: Annotated[Optional[float], None, "The wind gusts at 10 meters above ground level."]

    daily_predictions: Annotated[List[WeatherPrediction], ..., "A list of daily weather predictions."]


class WeatherResponse(ToolResults):
    weather_data: WeatherData
