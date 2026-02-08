from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage


@dataclass
class KGraphWorker(ABC):
    """
    Base class for KGraph workers that can generate subgraphs for LangGraph.
    
    Each worker represents a specific type of processing capability that can be
    dynamically added to a LangGraph execution graph. Workers are responsible for:
    1. Defining their processing logic
    2. Generating their own subgraph structure
    3. Managing their internal state and execution flow
    """
    
    name: str
    llm: ChatOpenAI
    system_directive: str = ""
    required_inputs: Optional[List[str]] = None
    max_iters: int = 6
    
    def __post_init__(self):
        if self.required_inputs is None:
            self.required_inputs = []
    
    @abstractmethod
    def build_subgraph(self, graph_builder: StateGraph, occurrence_id: str) -> Tuple[str, str]:
        """
        Build and add this worker's subgraph to the provided StateGraph.
        
        Args:
            graph_builder: The StateGraph to add nodes and edges to
            occurrence_id: Unique identifier for this worker occurrence in the graph
            
        Returns:
            Tuple of (entry_node_id, exit_node_id) for connecting to other parts of the graph
        """
        pass
    
    def _safe_node_id(self, occurrence_id: str, *parts: str) -> str:
        """
        Generate a safe node ID by combining occurrence_id with additional parts.
        """
        import re
        combined = "__".join([occurrence_id] + list(parts))
        return re.sub(r"[^A-Za-z0-9_.-]", "_", combined)
    
    def _get_worker_slot(self, state: Dict[str, Any], occurrence_id: str) -> Dict[str, Any]:
        """
        Get or create the worker's slot.
        IMPORTANT: Does NOT mutate state — returns existing slot or a fresh dict.
        Callers must include the slot in their partial return via {"work": {occurrence_id: slot}}.
        """
        work = state.get("work", {})
        if occurrence_id in work:
            return work[occurrence_id]
        # Return a fresh slot (not attached to state)
        return {"iters": 0, "messages": []}
    
    def _get_activation(self, state: Dict[str, Any], occurrence_id: str) -> Dict[str, Any]:
        """
        Get the activation data for this worker occurrence.
        """
        # Standard location: state["agent_data"]["activation"]
        agent_data = state.get("agent_data", {})
        activation_map = agent_data.get("activation", {})
        return activation_map.get(occurrence_id, {"prompt": "", "args": {}})
    
    def _finalize_result(self, state: Dict[str, Any], occurrence_id: str, result_text: str) -> Dict[str, Any]:
        """
        Finalize the worker's result and clean up its working state.
        Returns a PARTIAL state update (only changed keys) for parallel-safe merging.
        """
        activation = state.get("agent_data", {}).get("activation", {}).get(occurrence_id, {})
        payload = {
            "result_text": result_text,
            "args_used": activation.get("args", {})
        }
        
        # Return partial update — reducers handle merging
        return {
            "agent_data": {
                "results": {occurrence_id: payload},
            }
        }
