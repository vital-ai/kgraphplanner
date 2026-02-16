from __future__ import annotations

import logging
from typing import Dict, Any, Tuple, List
from dataclasses import dataclass

from langgraph.graph import StateGraph
from langgraph.runtime import get_runtime
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from kgraphplanner.worker.kgraph_worker import KGraphWorker

logger = logging.getLogger(__name__)


@dataclass
class KGCase:
    """
    Represents a categorization case with id, name, and description.
    
    Example:
        KGCase(
            id="weather",
            name="Weather Topic", 
            description="a category for weather related requests"
        )
    """
    id: str
    name: str
    description: str
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description
        }
    
    def __str__(self) -> str:
        """String representation for prompts."""
        return f"ID: {self.id}, Name: {self.name}, Description: {self.description}"


@dataclass
class KGraphCaseWorker(KGraphWorker):
    """
    A categorization worker that classifies input text into predefined categories.
    
    This worker takes a list of KGCase objects and categorizes user input
    by selecting the most appropriate case based on the input content.
    
    The worker creates a subgraph with:
    - Entry node: case_node (performs LLM categorization)
    - Exit node: case_node (same node, single step)
    
    The worker takes the activation prompt, args, and a list of cases,
    makes an LLM call to categorize the input, and returns the selected case.
    """
    
    cases: List[KGCase] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.cases is None:
            self.cases = []
    
    def build_subgraph(self, graph_builder: StateGraph, occurrence_id: str) -> Tuple[str, str]:
        """
        Build a single-node subgraph for case categorization.
        
        Returns:
            Tuple of (entry_node_id, exit_node_id) - both are the same for this simple worker
        """
        case_node_id = self._safe_node_id(occurrence_id, "case")
        
        async def case_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """Perform categorization and return the selected case."""
            runtime = get_runtime()
            writer = runtime.stream_writer
            
            # Get activation data
            activation = self._get_activation(state, occurrence_id)
            prompt = activation.get("prompt", "")
            args = activation.get("args", {})
            
            # Get cases from args or use worker's default cases
            cases_to_use = args.get("cases", self.cases)
            if not cases_to_use:
                error_msg = "No cases provided for categorization"
                writer({
                    "phase": "case_error",
                    "node": occurrence_id,
                    "worker": self.name,
                    "error": error_msg
                })
                return self._finalize_result(state, occurrence_id, f"Error: {error_msg}")
            
            # Build messages for LLM
            # Resolve directive: args may override/append via bindings
            effective_directive = self._resolve_system_directive(args)
            messages = []
            if effective_directive:
                messages.append(SystemMessage(content=effective_directive))
            
            # Get the user input to categorize
            user_input = args.get("query", args.get("message", args.get("input", "")))
            if not user_input:
                error_msg = "No input provided for categorization"
                writer({
                    "phase": "case_error",
                    "node": occurrence_id,
                    "worker": self.name,
                    "error": error_msg
                })
                return self._finalize_result(state, occurrence_id, f"Error: {error_msg}")
            
            # Add categorization instructions
            case_descriptions = "\n".join([f"- {case}" for case in cases_to_use])
            categorization_prompt = f"""You are a categorization assistant. Your task is to classify the given input into one of the predefined categories.

Available Categories:
{case_descriptions}

Instructions:
1. Analyze the input text carefully
2. Select the most appropriate category based on the content
3. Respond with ONLY the category ID (e.g., "weather", "travel", etc.)
4. If no category fits well, respond with "unknown"

Please categorize this input: "{user_input}" """
            
            messages.append(SystemMessage(content=categorization_prompt))
            
            if prompt:
                messages.append(SystemMessage(content=f"Additional instructions: {prompt}"))
            
            # Add a human message to trigger response
            messages.append(HumanMessage(content=f"Categorize: {user_input}"))
            
            writer({
                "phase": "case_start",
                "node": occurrence_id,
                "worker": self.name,
                "activation": activation,
                "cases_count": len(cases_to_use)
            })
            
            # Log LLM parameters for diagnostics (works for both OpenAI and Anthropic)
            _llm_inner = getattr(self.llm, 'bound', self.llm)  # unwrap RunnableBinding
            _llm_params = {
                "model": getattr(_llm_inner, 'model_name', getattr(_llm_inner, 'model', '?')),
                "temperature": getattr(_llm_inner, 'temperature', '?'),
                "max_tokens": getattr(_llm_inner, 'max_tokens', 'None'),
                "n_messages": len(messages),
            }
            logger.info(f"Case worker '{occurrence_id}' LLM params: {_llm_params}")

            try:
                # Make LLM call
                response = await self.llm.ainvoke(messages)
                result_text = response.content if hasattr(response, 'content') else str(response)
                
                # Clean up the result (remove whitespace, make lowercase)
                selected_case_id = result_text.strip().lower()
                
                # Find the matching case
                selected_case = None
                for case in cases_to_use:
                    if case.id.lower() == selected_case_id:
                        selected_case = case
                        break
                
                if selected_case:
                    result_summary = f"Categorized as: {selected_case.name} ({selected_case.id})"
                    final_result = {
                        "selected_case_id": selected_case.id,
                        "selected_case_name": selected_case.name,
                        "selected_case_description": selected_case.description,
                        "result_text": result_summary,
                        "raw_response": result_text,
                        "input": user_input
                    }
                else:
                    result_summary = f"Categorized as: {selected_case_id} (no match found)"
                    final_result = {
                        "selected_case_id": selected_case_id,
                        "selected_case_name": "Unknown",
                        "selected_case_description": "No matching category found",
                        "result_text": result_summary,
                        "raw_response": result_text,
                        "input": user_input
                    }
                
                writer({
                    "phase": "case_complete",
                    "node": occurrence_id,
                    "worker": self.name,
                    "result": result_summary,
                    "selected_case": final_result
                })
                
                # Use _finalize_result but override the result format to store our dict
                # First call _finalize_result with a placeholder
                result_text = f"CASE_RESULT:{final_result['selected_case_id']}"
                finalized_state = self._finalize_result(state, occurrence_id, result_text)
                
                # Then override the result with our full dict
                finalized_state["agent_data"]["results"][occurrence_id] = final_result
                
                return finalized_state
                
            except Exception as e:
                writer({
                    "phase": "case_error",
                    "node": occurrence_id,
                    "worker": self.name,
                    "error": str(e)
                })
                
                # Store error and return
                error_result = {
                    "selected_case_id": "error",
                    "selected_case_name": "Error",
                    "selected_case_description": "Categorization failed",
                    "raw_response": f"Error: {str(e)}",
                    "input": user_input
                }
                
                # Use _finalize_result for consistency
                error_text = f"CASE_ERROR:{str(e)}"
                finalized_state = self._finalize_result(state, occurrence_id, error_text)
                finalized_state["agent_data"]["results"][occurrence_id] = error_result
                
                return finalized_state
        
        # Add the single node to the graph
        graph_builder.add_node(case_node_id, case_node)
        
        # Return the same node as both entry and exit
        return case_node_id, case_node_id