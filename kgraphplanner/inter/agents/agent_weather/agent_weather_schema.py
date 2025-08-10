from typing import Annotated, List
from vital_agent_kg_utils.vital_agent_rest_resource_client.tools.weather.weather_response import WeatherData
from kgraphplanner.inter.base_agent_schema import BaseAgentResponse, BaseAgentRequest


class AgentWeatherRequest(BaseAgentRequest):
    """
        Use this to get weather information for today and the next few days given the name of a place.
    """

    place_name_list: Annotated[List[str], ..., "The place name of the weather location"]


class AgentWeatherResponse(BaseAgentResponse):
    """Represents a daily weather prediction."""

    place_name: Annotated[str, ..., "The place name of the weather location"]
    weatherReportText: Annotated[str, ..., "Text giving the weather report."]
    weatherReport: Annotated["WeatherData", ..., "Structured data of the weather report."]


class GenerateAgentWeatherSchema:
    pass
    # return schema and string version of schema for request and response
    # use temporarily for testing until replaced with dynamically generated
    # version for generating based on declared input/output of agent
    # as defined by input/output messages and KG frames
    # need to dynamically generate the Union of the request classes to include them to the LLM call:
    # agent_request: Union[AgentWeatherRequest, BaseAgentRequest] = Field(..., description="The details of the Agent request which must be an instance of a subclass of BaseAgentRequest")
