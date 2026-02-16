import logging
import contextvars
from typing import Dict, List, Optional, Any
from pathlib import Path

# Request-scoped JWT token — set per async task / request context.
# Falls back to the instance-level token when not set.
_request_jwt: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar('_request_jwt', default=None)

logger = logging.getLogger(__name__)

from kgraphplanner.config.agent_config import AgentConfig
from kgraphplanner.tool_manager.tool_inf import AbstractTool
from kgraphplanner.vital_agent_rest_resource_client.tools.tool_name_enum import ToolName as ToolNameEnum


class ToolManager:
    """Manages tools for KGraphPlanner agents."""
    
    def __init__(self, *, config: Optional[AgentConfig] = None):
        """
        Initialize the tool manager.
        
        Args:
            config: An AgentConfig instance. If not provided, uses defaults.
        """
        self.config = config if config is not None else AgentConfig()
        self.tools: Dict[str, AbstractTool] = {}
        self._tool_functions = {}
        self._jwt_token: Optional[str] = None
    
    def add_tool(self, tool: AbstractTool) -> None:
        """
        Add a tool to the manager.
        
        Args:
            tool: Tool instance to add
        """
        tool_name = tool.get_tool_name()
        self.tools[tool_name] = tool
        self._tool_functions[tool_name] = tool.get_tool_function()
    
    def get_tool(self, tool_name: str) -> Optional[AbstractTool]:
        """
        Get a tool by name.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool instance or None if not found
        """
        return self.tools.get(tool_name)
    
    def get_tool_function(self, tool_name: str):
        """
        Get a tool function by name.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool function or None if not found
        """
        return self._tool_functions.get(tool_name)
    
    def get_all_tools(self) -> Dict[str, AbstractTool]:
        """Get all registered tools."""
        return self.tools.copy()
    
    def get_all_tool_functions(self) -> List:
        """Get all tool functions for use with LangChain agents."""
        return list(self._tool_functions.values())
    
    def get_enabled_tools(self) -> List[AbstractTool]:
        """Get only the enabled tools based on configuration."""
        enabled_tool_names = self.config.get_enabled_tools()
        
        enabled_tools = []
        for tool_name in enabled_tool_names:
            if tool_name in self.tools:
                enabled_tools.append(self.tools[tool_name])
        
        return enabled_tools
    
    def get_enabled_tool_functions(self) -> List:
        """Get tool functions for enabled tools only."""
        enabled_tools = self.get_enabled_tools()
        return [tool.get_tool_function() for tool in enabled_tools]
    
    def load_tools_from_config(self) -> None:
        """Load and initialize tools based on configuration."""
        tool_config = self.config.get_tool_config()
        
        # Create tool configuration for individual tools
        individual_tool_config = {
            "tool_endpoint": self.config.get_tool_endpoint()
        }
        
        # Import and initialize available tools based on enabled list
        enabled_tool_names = self.config.get_enabled_tools()
        
        for tool_name in enabled_tool_names:
            try:
                if tool_name == ToolNameEnum.google_web_search_tool.value:
                    from kgraphplanner.tools.websearch.web_search_tool import WebSearchTool
                    websearch_tool = WebSearchTool(individual_tool_config, self)
                elif tool_name == ToolNameEnum.place_search_tool.value:
                    from kgraphplanner.tools.place_search.place_search_tool import PlaceSearchTool
                    place_search_tool = PlaceSearchTool(individual_tool_config, self)
                elif tool_name == ToolNameEnum.google_address_validation_tool.value:
                    from kgraphplanner.tools.address_validation.address_validation_tool import AddressValidationTool
                    address_validation_tool = AddressValidationTool(individual_tool_config, self)
                elif tool_name == ToolNameEnum.weather_tool.value:
                    from kgraphplanner.tools.weather.weather_tool import WeatherTool
                    weather_tool = WeatherTool(individual_tool_config, self)
                elif tool_name == "knowledge_search_tool":
                    from kgraphplanner.tools.knowledge.knowledge_search_tool import KnowledgeSearchTool
                    knowledge_tool = KnowledgeSearchTool(individual_tool_config, self)
            except ImportError as e:
                logger.warning(f"Could not load tool '{tool_name}': {e}")
    
    def list_available_tools(self) -> List[str]:
        """List all available tool names. Alias for :meth:`get_tool_names`."""
        return self.get_tool_names()
    
    def get_tool_descriptions(self) -> Dict[str, str]:
        """Get descriptions for all tools."""
        return {name: tool.get_tool_description() for name, tool in self.tools.items()}
    
    def get_tool_names(self) -> List[str]:
        """Get all available tool names."""
        return list(self.tools.keys())
    
    def set_jwt_token(self, jwt_token: str) -> None:
        """
        Set the JWT token for use with vital resource REST service clients.
        
        When running inside an async event loop this sets a request-scoped
        token via ``contextvars`` so concurrent requests don't overwrite
        each other.  The instance-level ``_jwt_token`` is kept as a
        fallback for sync callers.
        
        Args:
            jwt_token: JWT token string to use for authentication
        """
        _request_jwt.set(jwt_token)
        self._jwt_token = jwt_token
    
    def get_jwt_token(self) -> Optional[str]:
        """
        Get the current JWT token.
        
        Prefers the request-scoped contextvar token (safe for concurrent
        async requests) and falls back to the instance-level token.
        
        Returns:
            JWT token string or None if not set
        """
        return _request_jwt.get() or self._jwt_token
    
    def clear_jwt_token(self) -> None:
        """Clear the current JWT token (both contextvar and instance)."""
        _request_jwt.set(None)
        self._jwt_token = None