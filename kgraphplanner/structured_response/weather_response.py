from typing import Annotated
from vital_agent_kg_utils.vital_agent_rest_resource_client.tools.weather.weather_response import WeatherData
from kgraphplanner.structured_response.structured_response import StructuredResponse


class WeatherResponse(StructuredResponse):
    """Represents a daily weather prediction."""

    weatherReportText: Annotated[str, ..., "Text giving the weather report."]
    weatherReport: Annotated["WeatherData", ..., "Structured data of the weather report."]

