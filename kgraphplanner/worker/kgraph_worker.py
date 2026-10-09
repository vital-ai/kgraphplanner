from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, ClassVar, Union
from dataclasses import dataclass, field

from langgraph.graph import StateGraph
from langgraph.types import interrupt
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage, SystemMessage

DIRECTIVE_KEYS = ("system_directive", "system_directive_append")


def is_present(value: Any) -> bool:
    """False for None, an empty or whitespace-only string, or an empty
    collection. Anything else is present, including 0 and False."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) > 0
    return True

# A system directive is a string, or a list of segments for prompt caching:
#   [{"text": "<stable instructions>", "cache": True}, {"text": "<per-run part>"}]
# Cached segments get an Anthropic cache_control marker when the worker's LLM
# is a ChatAnthropic model; for other models the segments are joined as text.
Directive = Union[str, List[Dict[str, Any]]]


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
    system_directive: Directive = ""
    required_inputs: Optional[List[str]] = None
    max_iters: int = 6
    # Logging and data retention. All off by default: request and response
    # content is never logged, streamed, or copied into results unless asked.
    log_content: bool = False      # log prompts, args and responses (DEBUG/INFO)
    stream_payloads: bool = False  # include content in custom stream events
    record_args: bool = False      # keep a copy of the activation args as results[node]["args_used"]
    # What this worker does, for KGraphPlannerAgent's prompt. When any
    # registered worker has one, the planner lists "name: description" lines.
    # Keyword-only, so subclass fields keep their 0.2.0 positional order.
    description: str = field(default="", kw_only=True)

    # False for workers that never call an LLM (e.g. KGraphFunctionWorker).
    uses_llm: ClassVar[bool] = True
    
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
    
    def _split_directive(self, args: Dict[str, Any]) -> Tuple[Directive, Dict[str, Any]]:
        """Resolve the effective system directive for this invocation.

        If the activation *args* contain a ``system_directive`` key its
        value **replaces** the worker's static :pyattr:`system_directive`.
        If a ``system_directive_append`` key is present its value is
        **appended** to the effective directive (separated by two
        newlines).

        Returns ``(directive, args_without_directive_keys)``. *args* itself is
        not modified: it is the activation held in graph state, and a loop that
        re-enters the node must still see the directive.
        """
        directive: Directive = self.system_directive

        override = args.get("system_directive")
        if override:
            directive = override if isinstance(override, list) else str(override)

        append = args.get("system_directive_append")
        if append:
            if isinstance(directive, list):
                directive = list(directive) + [{"text": str(append)}]
            elif directive:
                directive = f"{directive}\n\n{str(append)}"
            else:
                directive = str(append)

        clean = {k: v for k, v in args.items() if k not in DIRECTIVE_KEYS}
        return directive, clean

    def _resolve_system_directive(self, args: Dict[str, Any]) -> Directive:
        """The effective directive for *args* (see ``_split_directive``).
        Does not modify *args*."""
        return self._split_directive(args)[0]

    def _directive_messages(self, directive: Directive) -> List[SystemMessage]:
        """System message(s) for a directive. Segmented directives get Anthropic
        cache_control markers on cached segments when the LLM supports them."""
        if not directive:
            return []
        if isinstance(directive, str):
            return [SystemMessage(content=directive)]
        segments = [seg for seg in directive if seg.get("text")]
        if not segments:
            return []
        inner = getattr(self.llm, "bound", self.llm)
        if "Anthropic" in type(inner).__name__:
            blocks = []
            for seg in segments:
                block: Dict[str, Any] = {"type": "text", "text": seg["text"]}
                if seg.get("cache"):
                    block["cache_control"] = {"type": "ephemeral"}
                blocks.append(block)
            return [SystemMessage(content=blocks)]
        return [SystemMessage(content="\n\n".join(seg["text"] for seg in segments))]

    @staticmethod
    def _first_input(args: Dict[str, Any], keys: Tuple[str, ...]) -> Any:
        """The first of ``keys`` whose value in *args* is non-empty, else "".

        Empty follows ``is_present`` (the presence gate's rule): None, a blank
        string or an empty collection is skipped, so ``{"query": [], "input":
        "hi"}`` gives "hi". 0 and False count as present.
        """
        for key in keys:
            value = args.get(key)
            if is_present(value):
                return value
        return ""

    def _is_activated(self, state: Dict[str, Any], occurrence_id: str) -> bool:
        """True if this occurrence has an activation entry.

        Some agents (e.g. ``KGraphCaseAgent``) wire every candidate worker and
        activate only the selected one; the others must do nothing. The exec
        graph agent always activates a node before it runs, so there a node
        with an entry but no prompt or args is an error, not a skip.
        """
        return occurrence_id in ((state.get("agent_data") or {}).get("activation") or {})

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
        payload: Dict[str, Any] = {"result_text": result_text}
        if self.record_args:
            activation = state.get("agent_data", {}).get("activation", {}).get(occurrence_id, {})
            payload["args_used"] = activation.get("args", {})
        
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
