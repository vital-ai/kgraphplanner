from typing import List, Type, Tuple
import logging

import httpx

from kgraphplanner.vital_agent_rest_resource_client.tool_service_interface import ToolServiceInterface
from kgraphplanner.vital_agent_rest_resource_client.tools.place_search.tool_handler import \
    PlaceSearchToolHandler
from kgraphplanner.vital_agent_rest_resource_client.tools.tool_parameters import ToolParameters
from kgraphplanner.vital_agent_rest_resource_client.tools.tool_request import ToolRequest
from kgraphplanner.vital_agent_rest_resource_client.tools.tool_response import ToolResponse
from kgraphplanner.vital_agent_rest_resource_client.tools.tool_results import ToolResults
from kgraphplanner.vital_agent_rest_resource_client.tools.weather.models import WeatherOutput, WeatherData
from kgraphplanner.vital_agent_rest_resource_client.tools.weather.tool_handler import WeatherToolHandler
from kgraphplanner.vital_agent_rest_resource_client.tools.web_search.tool_handler import WebSearchToolHandler


# client to the python webservice which provides access to tools
# and queries to kgraph service (graph and vector db)


class VitalAgentRestResourceClient(ToolServiceInterface):

    def __init__(self, config:dict, jwt_token: str = None):
        self.config = config
        self.jwt_token = jwt_token
        self.logger = logging.getLogger(__name__)

    # Tools

    # ── helpers ────────────────────────────────────────────────────────

    def _build_request(self, tool_name: str, tool_parameters: ToolParameters):
        """Build URL, payload dict, and headers for a tool request."""
        tool_request = ToolRequest(tool=tool_name, tool_input=tool_parameters)
        tool_endpoint = self.config.get("tool_endpoint")
        url = f"{tool_endpoint}/tool"
        payload = tool_request.to_dict()
        headers = {"Content-Type": "application/json"}
        if self.jwt_token:
            headers["Authorization"] = f"Bearer {self.jwt_token}"
        return url, payload, headers

    def _parse_tool_response(self, tool_name: str, tool_parameters: ToolParameters, response_json: dict) -> ToolResponse:
        """Parse the JSON response from the tool server into a ToolResponse."""

        if tool_name == "weather_tool":
            handler = WeatherToolHandler()
            return ToolResponse.create_success(
                tool_output=handler.handle_response(tool_parameters, response_json),
                duration_ms=0,
            )

        if tool_name == "place_search_tool":
            handler = PlaceSearchToolHandler()
            return ToolResponse.create_success(
                tool_output=handler.handle_response(tool_parameters, response_json),
                duration_ms=0,
            )

        if tool_name == "google_address_validation_tool":
            from kgraphplanner.vital_agent_rest_resource_client.tools.google_address_validation.tool_handler import AddressValidationToolHandler
            handler = AddressValidationToolHandler()
            return ToolResponse.create_success(
                tool_output=handler.handle_response(tool_parameters, response_json),
                duration_ms=0,
            )

        if tool_name == "google_web_search_tool":
            handler = WebSearchToolHandler()
            return ToolResponse.create_success(
                tool_output=handler.handle_response(tool_parameters, response_json),
                duration_ms=0,
            )

        # unknown tool
        return ToolResponse(
            tool_name=tool_name,
            tool_parameters=tool_parameters,
            tool_results=ToolResults(),
        )

    async def handle_tool_request(self, tool_name: str, tool_parameters: ToolParameters) -> ToolResponse:
        url, payload, headers = self._build_request(tool_name, tool_parameters)

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload, headers=headers)

        self.logger.info(f"Tool request to {url} - Status Code: {response.status_code}")
        response_json = response.json()
        self.logger.debug(f"Response JSON: {response_json}")

        return self._parse_tool_response(tool_name, tool_parameters, response_json)
