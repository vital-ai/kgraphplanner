from abc import ABC, abstractmethod
from typing import Callable, Type, Any, Dict, Optional
from pydantic import BaseModel


class AbstractTool(ABC):
    """Abstract base class for all tool implementations."""
    
    def __init__(self, 
                 config: Dict[str, Any],
                 tool_manager: Optional[Any] = None,
                 name: Optional[str] = None,
                 description: Optional[str] = None):
        """
        Initialize the abstract tool.
        
        Args:
            config: Configuration dictionary for the tool
            tool_manager: Reference to the tool manager (optional)
            name: Custom name for the tool (defaults to class name)
            description: Custom description for the tool (defaults to name)
        """
        self.config = config
        self.tool_manager = tool_manager
        
        if name:
            self.name = name
        else:
            self.name = self.__class__.__name__
            
        if description:
            self.description = description
        else:
            self.description = self.name
            
        # Register with tool manager if provided (after name is set)
        if self.tool_manager:
            self.tool_manager.add_tool(self)

    def get_tool_name(self) -> str:
        """Get the name of the tool."""
        return self.name

    def get_tool_description(self) -> str:
        """Get the description of the tool."""
        return self.description

    @abstractmethod
    def get_tool_function(self) -> Callable:
        """
        Get the LangChain tool function for this tool.
        
        Returns:
            Callable: The tool function decorated with @tool
        """
        pass

    @abstractmethod
    def get_tool_schema(self) -> Type[BaseModel]:
        """
        Get the Pydantic schema for this tool's parameters.
        
        Returns:
            Type[BaseModel]: The Pydantic model class for parameters
        """
        pass
