import logging
from typing import Callable, Type

logger = logging.getLogger(__name__)

from pydantic import BaseModel
from langchain_core.tools import tool

from kgraphplanner.vital_agent_rest_resource_client.tools.weather.models import (
    WeatherInput, 
    WeatherOutput
)
from kgraphplanner.vital_agent_rest_resource_client.vital_agent_rest_resource_client import (
    VitalAgentRestResourceClient
)
from kgraphplanner.vital_agent_rest_resource_client.tools.tool_name_enum import ToolName as ToolNameEnum

from kgraphplanner.tool_manager.tool_inf import AbstractTool


class WeatherTool(AbstractTool):
    """Weather tool implementation using VitalAgentRestResourceClient."""
    
    def __init__(self, config, tool_manager=None):
        super().__init__(
            config=config,
            tool_manager=tool_manager,
            name=ToolNameEnum.weather_tool.value,
            description="Get current weather information for a location"
        )
    
    def get_tool_schema(self) -> Type[BaseModel]:
        """Get the Pydantic schema for weather parameters."""
        return WeatherInput
    
    def get_tool_function(self) -> Callable:
        """Get the tool function for weather."""
        
        @tool(args_schema=WeatherInput)
        async def weather_tool(latitude: float, longitude: float, place_label: str = "", 
                        include_previous: bool = False, use_archive: bool = False, 
                        archive_date: str = "") -> WeatherOutput:
            """
            Get current weather information for a location.
            
            Args:
                latitude: Latitude coordinate
                longitude: Longitude coordinate  
                place_label: Label for the place (optional)
                include_previous: Include previous weather data
                use_archive: Use archived weather data
                archive_date: Archive date if using archived data
                
            Returns:
                WeatherOutput: Weather information results
            """
            
            # Create WeatherInput from parameters
            weather_input = WeatherInput(
                latitude=latitude,
                longitude=longitude,
                place_label=place_label,
                include_previous=include_previous,
                use_archive=use_archive,
                archive_date=archive_date
            )
            
            # Get tool endpoint from config
            tool_endpoint = self.config.get("tool_endpoint")
            if not tool_endpoint:
                # Return error in WeatherOutput format for consistency
                from kgraphplanner.vital_agent_rest_resource_client.tools.weather.models import WeatherData
                error_weather_data = WeatherData(
                    latitude=latitude,
                    longitude=longitude,
                    timezone="Unknown",
                    current={},
                    daily={}
                )
                return WeatherOutput(
                    tool="weather_tool",
                    weather_data=error_weather_data
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
                # Execute the weather request (async)
                tool_response = await client.handle_tool_request(ToolNameEnum.weather_tool.value, weather_input)
                
                # Extract results - should be WeatherOutput
                weather_results: WeatherOutput = tool_response.tool_output
                
                return weather_results
                
            except Exception as e:
                logger.warning(f"Weather tool error: {e}")
                from kgraphplanner.vital_agent_rest_resource_client.tools.weather.models import WeatherData
                error_weather_data = WeatherData(
                    latitude=latitude,
                    longitude=longitude,
                    timezone="Unknown",
                    current={},
                    daily={}
                )
                return WeatherOutput(
                    tool="weather_tool",
                    weather_data=error_weather_data
                )
        
        return weather_tool
    
    def get_weather(self, place_label: str, latitude: float, longitude: float, 
                   include_previous: bool = False, use_archive: bool = False, 
                   archive_date: str = "") -> WeatherOutput:
        """
        Direct method to get weather without going through tool function.
        
        Args:
            place_label: Label for the place
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            include_previous: Include previous weather data
            use_archive: Use archived weather data
            archive_date: Archive date if using archived data
            
        Returns:
            WeatherOutput: Weather information results
        """
        # Create WeatherInput from parameters
        weather_input = WeatherInput(
            place_label=place_label,
            latitude=latitude,
            longitude=longitude,
            include_previous=include_previous,
            use_archive=use_archive,
            archive_date=archive_date
        )
        
        tool_function = self.get_tool_function()
        return tool_function.invoke(weather_input)
