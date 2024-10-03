from typing_extensions import TypedDict
from kgraphplanner.tools.weather.weather_info_tool import WeatherReport


class WeatherResponse(TypedDict):

    weatherReportText: str

    weatherReport: WeatherReport

    weatherResponseNotes: str

    toolResponseList: list[TypedDict]

