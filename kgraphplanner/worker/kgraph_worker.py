from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from langgraph.graph import StateGraph
from langgraph.types import interrupt
from langchain_core.language_models.chat_models import BaseChatModel
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
    llm: BaseChatModel
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
    
    def _resolve_system_directive(self, args: Dict[str, Any]) -> str:
        """Resolve the effective system directive for this invocation.

        If the activation *args* contain a ``system_directive`` key its
        value **replaces** the worker's static :pyattr:`system_directive`.
        If a ``system_directive_append`` key is present its value is
        **appended** to the effective directive (separated by two
        newlines).  Both keys are consumed (removed from *args*) so they
        are not forwarded to the LLM as regular arguments.

        When neither key is present the worker's static directive is
        returned unchanged — fully backward-compatible.
        """
        directive = self.system_directive

        override = args.pop("system_directive", None)
        if override:
            directive = str(override)

        append = args.pop("system_directive_append", None)
        if append:
            if directive:
                directive = f"{directive}\n\n{str(append)}"
            else:
                directive = str(append)

        return directive

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
    
    def request_human_input(
        self,
        question: str,
        *,
        context: Optional[Dict[str, Any]] = None,
        options: Optional[List[str]] = None,
    ) -> Any:
        """
        Pause the graph and ask the human for input.
        
        Calls LangGraph's interrupt() which serializes the graph state and
        returns the payload to the caller.  When the caller invokes
        Command(resume=value), interrupt() returns that value here and
        execution continues from this point.
        
        Args:
            question: The question to present to the user.
            context: Optional dict of context (e.g., what failed, what was tried).
            options: Optional list of suggested responses.
            
        Returns:
            The user's reply when the graph resumes via Command(resume=...).
        """
        payload: Dict[str, Any] = {
            "type": "human_input_request",
            "worker": self.name,
            "question": question,
        }
        if context:
            payload["context"] = context
        if options:
            payload["options"] = options
        
        return interrupt(payload)
