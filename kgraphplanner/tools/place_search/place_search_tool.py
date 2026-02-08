import logging
from typing import Callable, Type

logger = logging.getLogger(__name__)

from pydantic import BaseModel
from langchain_core.tools import tool

from vital_agent_kg_utils.vital_agent_rest_resource_client.tools.place_search.models import (
    PlaceSearchInput, 
    PlaceSearchOutput
)
from vital_agent_kg_utils.vital_agent_rest_resource_client.vital_agent_rest_resource_client import (
    VitalAgentRestResourceClient
)
from vital_agent_kg_utils.vital_agent_rest_resource_client.tools.tool_name_enum import ToolName as ToolNameEnum

from kgraphplanner.tool_manager.tool_inf import AbstractTool


class PlaceSearchTool(AbstractTool):
    """Place search tool implementation using VitalAgentRestResourceClient."""
    
    def __init__(self, config, tool_manager=None):
        super().__init__(
            config=config,
            tool_manager=tool_manager,
            name=ToolNameEnum.place_search_tool.value,
            description="Search for places and get detailed location information"
        )
    
    def get_tool_schema(self) -> Type[BaseModel]:
        """Get the Pydantic schema for place search parameters."""
        return PlaceSearchInput
    
    def get_tool_function(self) -> Callable:
        """Get the tool function for place search."""
        
        @tool(args_schema=PlaceSearchInput)
        def place_search_tool(place_search_string: str) -> PlaceSearchOutput:
            """
            Search for places and get detailed location information.
            
            Args:
                place_search_string: The place search query string
                
            Returns:
                PlaceSearchOutput: Place search results with details
            """
            
            # Create PlaceSearchInput from parameter
            place_search_input = PlaceSearchInput(place_search_string=place_search_string)
            
            # Get tool endpoint from config
            tool_endpoint = self.config.get("tool_endpoint")
            if not tool_endpoint:
                # Return error in PlaceSearchOutput format for consistency
                return PlaceSearchOutput(
                    results=[]
                )
            
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
                # Execute the search
                tool_response = client.handle_tool_request(ToolNameEnum.place_search_tool.value, place_search_input)
                
                # Extract results - should be PlaceSearchOutput
                place_search_results: PlaceSearchOutput = tool_response.tool_output
                
                return place_search_results
                
            except Exception as e:
                logger.warning(f"Place search tool error: {e}")
                error_output = PlaceSearchOutput(
                    results=[]
                )
                return error_output
        
        return place_search_tool
    
    def execute_search(self, place_search_string: str) -> PlaceSearchOutput:
        """
        Direct method to execute place search without going through tool function.
        
        Args:
            place_search_string: The place search string
            
        Returns:
            PlaceSearchOutput: Place search results with details
        """
        # Create PlaceSearchInput from parameters
        place_search_input = PlaceSearchInput(
            place_search_string=place_search_string
        )
        
        tool_function = self.get_tool_function()
        return tool_function.invoke(place_search_input)
