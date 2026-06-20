import logging
import traceback
from typing import Callable, Type

logger = logging.getLogger(__name__)

from pydantic import BaseModel
from langchain_core.tools import tool

from kgraphplanner.vital_agent_rest_resource_client.tools.serper_web_search.models import (
    SerperWebSearchInput,
    SerperWebSearchOutput
)
from kgraphplanner.vital_agent_rest_resource_client.vital_agent_rest_resource_client import (
    VitalAgentRestResourceClient
)
from kgraphplanner.vital_agent_rest_resource_client.tools.tool_name_enum import ToolName as ToolNameEnum

from kgraphplanner.tool_manager.tool_inf import AbstractTool


class SerperWebSearchTool(AbstractTool):
    """Serper web search tool implementation using VitalAgentRestResourceClient."""

    def __init__(self, config, tool_manager=None):
        super().__init__(
            config=config,
            tool_manager=tool_manager,
            name=ToolNameEnum.serper_web_search_tool.value,
            description="Search the web using Serper and return relevant results including news, images, shopping, and places"
        )

    def get_tool_schema(self) -> Type[BaseModel]:
        """Get the Pydantic schema for Serper web search parameters."""
        return SerperWebSearchInput

    def get_tool_function(self) -> Callable:
        """Get the tool function for Serper web search."""

        @tool(args_schema=SerperWebSearchInput)
        async def serper_web_search_tool(search_query: str, num_results: int = 10,
                                         search_type: str = "search", location: str = None,
                                         time_period: str = None) -> SerperWebSearchOutput:
            """
            Search the web using Serper and return relevant results.

            Args:
                search_query: The search query string
                num_results: Number of search results to return (default: 10)
                search_type: Type of search (search/news/images/shopping/places)
                location: Location for localized search results (e.g., 'New York,New York')
                time_period: Time period filter (hour/day/week/month/year)

            Returns:
                SerperWebSearchOutput: Search results with titles, URLs, snippets, and enriched data
            """

            serper_input = SerperWebSearchInput(
                search_query=search_query,
                num_results=num_results,
                search_type=search_type,
                location=location,
                time_period=time_period
            )

            tool_endpoint = self.config.get("tool_endpoint")
            if not tool_endpoint:
                return "No search results available: tool endpoint is not configured."

            client_config = {
                "tool_endpoint": tool_endpoint
            }

            jwt_token = None
            if self.tool_manager:
                jwt_token = self.tool_manager.get_jwt_token()

            client = VitalAgentRestResourceClient(client_config, jwt_token)

            try:
                tool_response = await client.handle_tool_request(ToolNameEnum.serper_web_search_tool.value, serper_input)

                if tool_response is None or not tool_response.success or tool_response.tool_output is None:
                    logger.warning(f"Serper search tool returned None response for query: {search_query}")
                    return f"No search results available for query: {search_query}"

                serper_results: SerperWebSearchOutput = tool_response.tool_output

                results = getattr(serper_results, 'results', None) or []
                if not results:
                    logger.info(f"Serper search returned empty results for query: {search_query}")
                    return f"Search returned no results for query: {search_query}"

                return serper_results

            except Exception as e:
                logger.warning(f"Serper search tool error ({type(e).__name__}): {e}")
                logger.warning(f"Serper search tool traceback:\n{traceback.format_exc()}")
                return f"No search results available for query: {search_query} (error: {type(e).__name__}: {e})"

        return serper_web_search_tool
