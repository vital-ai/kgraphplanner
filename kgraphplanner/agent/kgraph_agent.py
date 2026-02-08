from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig


class KGraphAgent(ABC):
    """
    Abstract interface for all KGraph agents.
    
    This interface defines the contract that all KGraph agents must implement,
    providing a consistent API for agent interaction and execution.
    """
    
    @abstractmethod
    async def arun(self, messages: List[BaseMessage], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """
        Asynchronously run the agent with the given messages.
        
        Args:
            messages: List of messages to process
            config: Optional configuration for the run
            
        Returns:
            Dictionary containing the agent's response and any additional data
        """
        pass
    
    @abstractmethod
    def get_compiled_graph(self):
        """
        Get the compiled LangGraph for this agent.
        
        Returns:
            The compiled graph that can be executed
        """
        pass
    
    @abstractmethod
    def get_agent_info(self) -> Dict[str, Any]:
        """
        Get information about this agent.
        
        Returns:
            Dictionary containing agent metadata (name, type, capabilities, etc.)
        """
        pass
