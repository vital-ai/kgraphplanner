import logging
import traceback
from typing import Callable, Type
import json

logger = logging.getLogger(__name__)

from pydantic import BaseModel, Field
from langchain_core.tools import tool, ToolException

from kgraphplanner.tools.tool_errors import check_api_error, fail, reports_failures

from kgraphplanner.vital_agent_rest_resource_client.tools.web_search.models import (
    WebSearchInput, 
    WebSearchOutput
)
from kgraphplanner.vital_agent_rest_resource_client.vital_agent_rest_resource_client import (
    VitalAgentRestResourceClient
)
from kgraphplanner.vital_agent_rest_resource_client.tools.tool_name_enum import ToolName as ToolNameEnum

from kgraphplanner.tool_manager.tool_inf import AbstractTool


class WebSearchTool(AbstractTool):
    """Web search tool implementation using VitalAgentRestResourceClient."""
    
    def __init__(self, config, tool_manager=None):
        super().__init__(
            config=config,
            tool_manager=tool_manager,
            name=ToolNameEnum.google_web_search_tool.value,
            description="Search the web using Google and return relevant results"
        )
    
    def get_tool_schema(self) -> Type[BaseModel]:
        """Get the Pydantic schema for web search parameters."""
        return WebSearchInput
    
    def get_tool_function(self) -> Callable:
        """Get the tool function for web search."""
        
        @tool(args_schema=WebSearchInput)
        async def google_web_search_tool(search_query: str, num_results: int = 5, location: str = None, 
                                 language: str = None, country: str = None, device: str = "desktop",
                                 safe_search: str = None, search_type: str = "search", 
                                 time_period: str = None, ludocid: str = None,
                                 kgmid: str = None) -> WebSearchOutput:
            """
            Search the web using Google and return relevant results.
            
            Args:
                search_query: The search query string
                num_results: Number of search results to return (default: 5)
                location: Location for search results
                language: Language for search results
                country: Country for search results
                device: Device type (desktop/mobile)
                safe_search: Safe search setting
                search_type: Type of search (search/images/news/shopping/local)
                time_period: Time period filter
                ludocid: Google CID (customer identifier) of a place for deep-diving a specific Google Maps listing
                kgmid: Google Knowledge Graph ID (KGMID) for querying a specific Knowledge Graph entity
                
            Returns:
                WebSearchOutput: Search results with titles, URLs, and snippets
            """
            
            # Create WebSearchInput from parameters
            web_search_input = WebSearchInput(
                search_query=search_query,
                num_results=num_results,
                location=location,
                language=language,
                country=country,
                device=device,
                safe_search=safe_search,
                search_type=search_type,
                time_period=time_period,
                ludocid=ludocid,
                kgmid=kgmid
            )
            
            # Get tool endpoint from config
            tool_endpoint = self.config.get("tool_endpoint")
            if not tool_endpoint:
                fail("No web search results available: tool endpoint is not configured.")
            
            # Create client configuration
            client_config = {
                "tool_endpoint": tool_endpoint
            }
            
            # Get JWT token from tool manager if available
            jwt_token = None
            if self.tool_manager:
                jwt_token = self.tool_manager.get_jwt_token()
            
            # Initialize the client
            client = VitalAgentRestResourceClient(client_config, jwt_token)
            
            try:
                # Execute the search (async)
                tool_response = await client.handle_tool_request(ToolNameEnum.google_web_search_tool.value, web_search_input)
                
                if tool_response is None or not tool_response.success or tool_response.tool_output is None:
                    detail = getattr(tool_response, "error_message", None) if tool_response else None
                    logger.error(f"Web search tool failed for query '{search_query}': {detail}")
                    fail(f"No web search results available for query: {search_query}"
                         + (f" (error: {detail})" if detail else ""))
                
                # Extract results - should be WebSearchOutput
                web_search_results: WebSearchOutput = tool_response.tool_output

                # An upstream SerpAPI failure arrives as a successful response with
                # zero results and api_error set. Note the service deliberately
                # leaves api_error None for a string `error` value, which is its
                # "no results for this query" message -- a genuine zero result.
                check_api_error(web_search_results)
                
                results = getattr(web_search_results, 'results', None) or []
                if not results:
                    logger.info(f"Web search returned empty results for query: {search_query}")
                    return f"Web search returned no results for query: {search_query}"
                
                return web_search_results
                
            except ToolException:
                raise
            except Exception as e:
                logger.error(f"Web search tool error ({type(e).__name__}): {e}")
                logger.error(f"Web search tool traceback:\n{traceback.format_exc()}")
                fail(f"No web search results available for query: {search_query} (error: {type(e).__name__}: {e})")
        
        return reports_failures(google_web_search_tool)
    
    def execute_search(self, search_query: str, num_results: int = 5) -> WebSearchOutput:
        """
        Direct method to execute web search without going through tool function.
        
        Args:
            search_query: The search query to execute
            num_results: Number of search results to return (default: 5)
            
        Returns:
            WebSearchOutput: Search results with titles, URLs, and snippets
        """
        # Create WebSearchInput from parameters
        web_search_input = WebSearchInput(
            search_query=search_query,
            num_results=num_results
        )
        
        tool_function = self.get_tool_function()
        return tool_function.invoke(web_search_input)
