from __future__ import annotations

from typing import Dict, Any, List, Optional, TypedDict, Annotated
from abc import ABC, abstractmethod

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.base import BaseCheckpointSaver

from kgraphplanner.agent.kgraph_agent import KGraphAgent


REPLACE = "__replace__"
ADD = "__add__"
APPEND = "__append__"


def replace_value(value: Any) -> Dict[str, Any]:
    """Wrap an agent_data value so the reducer replaces the key instead of merging.

    Plain dict values are merged one level deep, so writing ``{}`` can never
    clear a key. Use ``{"activation": replace_value({})}`` to reset it.
    """
    return {REPLACE: value}


def add_values(value: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap a nested dict of numbers so the reducer adds it to the current value."""
    return {ADD: value}


def append_values(items: List[Any]) -> Dict[str, Any]:
    """Wrap a list so the reducer appends it to the current list."""
    return {APPEND: list(items)}


def _op(value: Any) -> Optional[str]:
    if isinstance(value, dict) and len(value) == 1:
        key = next(iter(value))
        if key in (REPLACE, ADD, APPEND):
            return key
    return None


def _add_nested(base: Any, inc: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base) if isinstance(base, dict) else {}
    for k, v in inc.items():
        if isinstance(v, dict):
            out[k] = _add_nested(out.get(k), v)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            prev = out.get(k, 0)
            out[k] = (prev if isinstance(prev, (int, float)) else 0) + v
        else:
            out[k] = v
    return out


def merge_agent_data(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """Merge agent_data dictionaries, with right values taking precedence.

    Dict values are merged one level deep (``results[node]`` is replaced as a
    whole). A value wrapped by ``replace_value``, ``add_values`` or
    ``append_values`` replaces, adds to, or appends to the current value.
    """
    if not right:
        return left or {}
    result = dict(left or {})

    for key, right_value in right.items():
        op = _op(right_value)
        if op == REPLACE:
            result[key] = right_value[REPLACE]
        elif op == ADD:
            result[key] = _add_nested(result.get(key), right_value[ADD])
        elif op == APPEND:
            prev = result.get(key)
            result[key] = (list(prev) if isinstance(prev, list) else []) + right_value[APPEND]
        elif key in result and isinstance(result[key], dict) and isinstance(right_value, dict):
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
