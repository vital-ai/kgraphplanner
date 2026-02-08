from __future__ import annotations

from typing import Dict, Any, List, Optional, TypedDict, Annotated
from abc import ABC, abstractmethod

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.base import BaseCheckpointSaver

from kgraphplanner.agent.kgraph_agent import KGraphAgent


def merge_agent_data(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """Merge agent_data dictionaries, with right values taking precedence."""
    if not left:
        return right or {}
    if not right:
        return left
    
    # Deep merge nested dictionaries
    result = dict(left)
    
    for key, right_value in right.items():
        if key in result and isinstance(result[key], dict) and isinstance(right_value, dict):
            # Deep merge nested dicts
            result[key] = {**result[key], **right_value}
        else:
            # Direct assignment for non-dict values or new keys
            result[key] = right_value
    
    return result


def merge_work(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """Merge work slot dicts. Each worker writes to work[occurrence_id]."""
    if not left:
        return right or {}
    if not right:
        return left
    return {**left, **right}


class AgentState(TypedDict):
    """Base state for KGraph agents"""
    messages: Annotated[List[BaseMessage], add_messages]
    agent_data: Annotated[Dict[str, Any], merge_agent_data]  # For agent-specific data with reducer
    work: Annotated[Dict[str, Any], merge_work]  # Per-worker execution slots (parallel-safe)


class KGraphBaseAgent(KGraphAgent, ABC):
    """
    Base class for all KGraph agents.
    
    Provides common functionality for building and executing LangGraph workflows
    using the worker architecture.
    """
    
    def __init__(self, name: str, checkpointer: Optional[BaseCheckpointSaver] = None):
        self.name = name
        self.checkpointer = checkpointer
        self._compiled_graph = None
    
    @abstractmethod
    def build_graph(self) -> StateGraph:
        """
        Build the LangGraph StateGraph for this agent.
        
        This method must be implemented by subclasses to define their specific
        graph structure and processing logic.
        
        Returns:
            StateGraph configured for this agent's workflow
        """
        pass
    
    def get_compiled_graph(self):
        """Get the compiled graph, building it if necessary."""
        if self._compiled_graph is None:
            graph = self.build_graph()
            self._compiled_graph = graph.compile(checkpointer=self.checkpointer)
        return self._compiled_graph
    
    async def arun(self, messages: List[BaseMessage], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """Run the agent with the given messages."""
        compiled_graph = self.get_compiled_graph()
        
        initial_state = {
            "messages": messages,
            "agent_data": {},
            "work": {}
        }
        
        result = await compiled_graph.ainvoke(initial_state, config=config)
        return result
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about this agent."""
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "has_checkpointer": self.checkpointer is not None
        }
