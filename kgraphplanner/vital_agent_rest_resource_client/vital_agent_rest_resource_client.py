from typing import List, Type, Tuple
import logging

import httpx

from kgraphplanner.tools.tool_logging import logged
from kgraphplanner.vital_agent_rest_resource_client.tool_service_interface import ToolServiceInterface
from kgraphplanner.vital_agent_rest_resource_client.tools.place_search.tool_handler import \
    PlaceSearchToolHandler
from kgraphplanner.vital_agent_rest_resource_client.tools.tool_parameters import ToolParameters
from kgraphplanner.vital_agent_rest_resource_client.tools.tool_request import ToolRequest
from kgraphplanner.vital_agent_rest_resource_client.tools.tool_response import ToolResponse
from kgraphplanner.vital_agent_rest_resource_client.tools.tool_results import ToolResults
from kgraphplanner.vital_agent_rest_resource_client.tools.weather.models import WeatherOutput, WeatherData
from kgraphplanner.vital_agent_rest_resource_client.tools.weather.tool_handler import WeatherToolHandler
from kgraphplanner.vital_agent_rest_resource_client.tools.serper_web_search.tool_handler import SerperWebSearchToolHandler
from kgraphplanner.vital_agent_rest_resource_client.tools.web_search.tool_handler import WebSearchToolHandler


# client to the python webservice which provides access to tools
# and queries to kgraph service (graph and vector db)


class VitalAgentRestResourceClient(ToolServiceInterface):

    def __init__(self, config:dict, jwt_token: str = None):
        self.config = config
        self.jwt_token = jwt_token
        self.logger = logging.getLogger(__name__)
        # From ToolConfig.log_content via the tool's client_config. Off:
        # service error bodies and raw responses stay out of the logs. The
        # error text returned to the caller (and on to the model) is unchanged.
        self.log_content = bool((config or {}).get("log_content", False))

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

        # The service reports a tool that caught its own exception as HTTP 200 with
        # success=false in the body, and `ToolResponse.to_dict()` there replaces the
        # null tool_output with {"results": []}. Every branch below calls
        # create_success unconditionally, so such a failure used to be stamped
        # successful and reach the agent as a well-formed empty result -- identical
        # to a search that genuinely matched nothing. The github handler has always
        # checked this (tools/github/tool_handler.py); the rest did not.
        if not response_json.get('success', True):
            msg = response_json.get('error_message') or f'{tool_name} reported failure'
            self.logger.error(f"Tool server reported failure for {tool_name}: "
                              f"{logged(msg, self.log_content, label='error message')}")
            return ToolResponse.create_error(error_message=msg, duration_ms=0)

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

        if tool_name == "serper_web_search_tool":
            handler = SerperWebSearchToolHandler()
            return ToolResponse.create_success(
                tool_output=handler.handle_response(tool_parameters, response_json),
                duration_ms=0,
            )

        # Registry-driven: one handler per github_* service tool, all sharing the
        # same parsing, so adding a service tool is one entry rather than a branch.
        from kgraphplanner.vital_agent_rest_resource_client.tools.github.tool_handler import (
            HANDLERS as GITHUB_HANDLERS
        )
        if tool_name in GITHUB_HANDLERS:
            handler = GITHUB_HANDLERS[tool_name]()
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

        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(url, json=payload, headers=headers)

        self.logger.info(f"Tool request to {url} - Status Code: {response.status_code}")

        if response.status_code in (401, 403):
            msg = f"Tool server auth error ({response.status_code}) for {tool_name}: {response.text}"
            self.logger.error(f"Tool server auth error ({response.status_code}) for {tool_name}: "
                              f"{logged(response.text, self.log_content, label='error body')}")
            raise PermissionError(msg)

        if response.status_code >= 400:
            msg = f"Tool server error ({response.status_code}) for {tool_name}: {response.text}"
            self.logger.error(f"Tool server error ({response.status_code}) for {tool_name}: "
                              f"{logged(response.text, self.log_content, label='error body')}")
            return ToolResponse.create_error(error_message=msg, duration_ms=0)

        response_json = response.json()
        if self.log_content:
            self.logger.debug(f"Response JSON: {response_json}")

        return self._parse_tool_response(tool_name, tool_parameters, response_json)
