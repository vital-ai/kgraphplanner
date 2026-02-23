
from __future__ import annotations

import logging
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from kgraphplanner.agent.kgraph_base_agent import KGraphBaseAgent, AgentState
from kgraphplanner.worker.kgraph_case_worker import KGraphCaseWorker, KGCase
from kgraphplanner.worker.kgraph_chat_worker import KGraphChatWorker
from kgraphplanner.worker.kgraph_worker import KGraphWorker


class KGraphCaseAgent(KGraphBaseAgent):
    """
    Agent that routes requests through case-based selection.
    
    Flow: user input -> case worker (selects case) -> selected worker -> resolve worker -> final result
    
    The agent takes a list of (KGCase, Worker) pairs, where each KGCase describes a category
    and each Worker handles that category. It uses an internal case worker to classify the input,
    then routes to the appropriate worker, and finally uses a resolve worker to clean up the output.
    """
    
    def __init__(
        self, 
        name: str, 
        case_worker_pairs: List[Tuple[KGCase, KGraphWorker]],
        llm: ChatOpenAI = None,
        checkpointer=None
    ):
        super().__init__(name, checkpointer)
        
        if not case_worker_pairs:
            raise ValueError("case_worker_pairs list cannot be empty")
        
        self.case_worker_pairs = case_worker_pairs
        self.cases = [pair[0] for pair in case_worker_pairs]
        self.workers = {pair[0].id: pair[1] for pair in case_worker_pairs}
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        
        # Create internal workers
        self._setup_internal_workers()
    
    def _setup_internal_workers(self):
        """Set up the internal case, and resolve workers."""
        
        # Use the KGCase objects directly from the pairs
        kg_cases = self.cases
        
        # Create case worker for classification
        self.case_worker = KGraphCaseWorker(
            name=f"{self.name}_case_worker",
            llm=self.llm,
            system_directive="You are a classification assistant that categorizes user requests into predefined categories.",
            cases=kg_cases,
            required_inputs=["input"]
        )
        
        # Create resolve worker for final output cleanup
        self.resolve_worker = KGraphChatWorker(
            name=f"{self.name}_resolve_worker", 
            llm=self.llm,
            system_directive="""You are a helpful assistant that creates friendly, polished responses. 
            Take the technical output from a worker and present it in a clear, user-friendly way. 
            Maintain all important information while making it conversational and easy to understand.""",
            required_inputs=["worker_output"]
        )
    
    def build_graph(self) -> StateGraph:
        """
        Build the case-based routing graph:
        entry -> case_worker -> worker_executor -> resolve_worker -> end
        """
        graph = StateGraph(AgentState)
        
        # Entry node - sets up activation for case worker ONLY
        def entry_node(state: AgentState) -> AgentState:
            messages = state.get("messages", [])
            user_input = messages[-1].content if messages else "Hello"
            
            # Use unique occurrence_id for case worker
            case_worker_occurrence_id = "case_classifier"
            
            logger.debug(f"Entry node - existing state keys: {list(state.get('agent_data', {}).keys())}")
            
            # Completely clear any existing state to prevent interference
            agent_data = {
                "activation": {
                    case_worker_occurrence_id: {
                        "prompt": "Classify the user input into the most appropriate category.",
                        "args": {
                            "input": user_input,
                            "cases": self.case_worker.cases
                        }
                    }
                    # NOTE: Only case worker is activated here, not all workers
                },
                "decisions": {},  # Clear any previous decisions
                "results": {},
                "errors": {},
                "work": {},
                "user_input": user_input,  # Store for later use
                "case_worker_occurrence_id": case_worker_occurrence_id  # Store for later lookup
            }
            
            logger.debug(f"Entry node - activation keys: {list(agent_data['activation'].keys())}")
            
            return {
                "messages": messages,
                "agent_data": agent_data
            }
        
        # Worker setup node - sets up activation for the selected worker
        def worker_setup_node(state: AgentState) -> AgentState:
            logger.debug(f"Worker setup node - state keys: {list(state.keys())}")
            agent_data = state.get("agent_data", {})
            results = agent_data.get("results", {})
            # Use the case worker's occurrence_id for result lookup
            case_worker_occurrence_id = agent_data.get("case_worker_occurrence_id", "case_classifier")
            case_result = results.get(case_worker_occurrence_id, {})
            
            logger.debug(f"Case result for '{case_worker_occurrence_id}': {case_result}")
            
            # Also check decisions in case the result is stored there
            decisions = agent_data.get("decisions", {})
            case_decision = decisions.get(case_worker_occurrence_id, {})
            logger.debug(f"Case decision for '{case_worker_occurrence_id}': {case_decision}")
            
            # Determine selected case - check both results and decisions
            selected_case = None  # Will default to unknown if no match found
            
            # Try to get case data from either results or decisions
            case_data = None
            if case_result:
                case_data = case_result
                logger.debug(f"Case data from results: {case_data}")
            elif case_decision:
                case_data = case_decision
                logger.debug(f"Case data from decisions: {case_decision}")
            
            if case_data:
                if isinstance(case_data, dict) and "selected_case_id" in case_data:
                    case_id = case_data.get("selected_case_id", "")
                    logger.debug(f"Selected case ID from dict: {case_id}")
                else:
                    case_id = str(case_data).strip().lower()
                    logger.debug(f"Selected case ID from string: {case_id}")
                
                # Find matching case by ID
                selected_worker = self.workers.get(case_id)
                if selected_worker:
                    # Find the corresponding case
                    for case in self.cases:
                        if case.id == case_id:
                            selected_case = case
                            logger.debug(f"Found matching case: {case.name}")
                            break
                else:
                    logger.debug(f"No worker found for case ID: {case_id}")
            
            # If no valid case found, check for UNKNOWN worker
            if selected_case is None:
                logger.debug(f"No valid case result found, checking for UNKNOWN worker")
                # Check if there's an UNKNOWN case worker pair
                unknown_worker = self.workers.get("UNKNOWN")
                if unknown_worker:
                    from kgraphplanner.worker.kgraph_case_worker import KGCase
                    selected_case = KGCase(id="UNKNOWN", name="Unknown", description="unclassifiable request")
                    logger.debug(f"Using UNKNOWN worker for unclassifiable request")
                else:
                    # No UNKNOWN worker provided - this is an error
                    error_msg = "Request could not be classified and no UNKNOWN worker is configured"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            
            # Set up activation only for the selected worker
            user_input = agent_data.get("user_input", "")
            
            # Clear ALL previous activation and decisions to prevent leftover state
            activation = {}  # Start fresh
            
            # Add activation only for the selected worker
            selected_worker = self.workers[selected_case.id]
            worker_occurrence_id = f"{selected_case.id}_worker"
            activation[worker_occurrence_id] = {
                "prompt": f"Handle this {selected_case.description} request.",
                "args": {
                    "request": user_input,
                    "message": user_input,
                    "query": user_input,
                    "input": user_input
                }
            }
            
            logger.debug(f"Worker setup - activating ONLY '{worker_occurrence_id}'")
            
            # Store selected case info for routing
            agent_data["selected_case"] = selected_case
            agent_data["selected_worker_id"] = worker_occurrence_id
            agent_data["activation"] = activation
            agent_data["decisions"] = {}  # Clear previous decisions
            agent_data["results"] = dict(agent_data.get("results", {}))  # Keep case worker result
            
            return {
                "messages": state.get("messages", []),
                "agent_data": agent_data
            }
        
        # Route to worker function - determines which worker to execute
        def route_to_worker(state: AgentState) -> str:
            agent_data = state.get("agent_data", {})
            selected_case = agent_data.get("selected_case")
            if selected_case:
                return f"{selected_case.id}_worker"
            # If no case selected, check for UNKNOWN worker
            if "UNKNOWN" in self.workers:
                return "UNKNOWN_worker"
            # No fallback available - this should not happen if worker_setup_node works correctly
            raise ValueError("No case selected and no UNKNOWN worker configured")
        
        # Resolve setup node - prepares activation for resolve worker
        def resolve_setup_node(state: AgentState) -> AgentState:
            agent_data = state.get("agent_data", {})
            selected_worker_id = agent_data.get("selected_worker_id", "")
            
            # Get the selected worker's result from decisions
            decisions = agent_data.get("decisions", {})
            worker_decision = decisions.get(selected_worker_id, {})
            
            logger.debug(f"Resolve setup - selected_worker_id: {selected_worker_id}, decisions: {list(decisions.keys())}")
            
            # Check if selected worker has a meaningful answer
            selected_answer = worker_decision.get("answer", "")
            has_meaningful_answer = (selected_answer and 
                                   selected_answer not in ["No activation data provided...", "None", None] and 
                                   len(selected_answer) > 50)
            
            if has_meaningful_answer:
                logger.debug(f"Selected worker '{selected_worker_id}' has meaningful answer")
                worker_output = selected_answer
            else:
                logger.debug(f"Selected worker '{selected_worker_id}' not ready or no meaningful answer")
                # Don't proceed with resolve setup if selected worker hasn't completed
                # Return current state to allow more processing
                return state
            
            logger.debug(f"Resolve setup - proceeding with worker_output: {str(worker_output)}")
            
            # Set up activation for resolve worker
            activation = agent_data.get("activation", {})
            activation.clear()  # Clear previous activations
            resolve_worker_occurrence_id = "resolve_worker"
            activation[resolve_worker_occurrence_id] = {
                "prompt": "Create a friendly, user-facing response from the worker output.",
                "args": {
                    "worker_output": str(worker_output),
                    "message": f"Please present this information in a friendly way: {worker_output}"
                }
            }
            
            agent_data["activation"] = activation
            
            return {
                "messages": state.get("messages", []),
                "agent_data": agent_data
            }
        
        # Add nodes
        graph.add_node("entry", entry_node)
        graph.add_node("worker_setup", worker_setup_node)
        graph.add_node("resolve_setup", resolve_setup_node)
        
        # Add worker subgraphs using occurrence_ids
        case_worker_occurrence_id = "case_classifier"
        case_entry, case_exit = self.case_worker.build_subgraph(graph, case_worker_occurrence_id)
        resolve_worker_occurrence_id = "resolve_worker"
        resolve_entry, resolve_exit = self.resolve_worker.build_subgraph(graph, resolve_worker_occurrence_id)
        
        # Connect the basic flow first
        graph.add_edge(START, "entry")
        graph.add_edge("entry", case_entry)
        graph.add_edge(case_exit, "worker_setup")
        
        # Add all worker subgraphs with static occurrence_ids
        worker_entries = {}
        worker_exits = {}
        for case in self.cases:
            worker = self.workers[case.id]
            worker_occurrence_id = f"{case.id}_worker"  # Static occurrence_id
            logger.debug(f"Adding worker subgraph for '{worker_occurrence_id}'")
            entry, exit = worker.build_subgraph(graph, worker_occurrence_id)
            worker_entries[worker_occurrence_id] = entry
            worker_exits[worker_occurrence_id] = exit
            logger.debug(f"Worker '{worker_occurrence_id}' entry: {entry}, exit: {exit}")
            
            # Add direct edge from worker_setup to worker entry
            # This ensures worker is never orphaned - conditional routing controls activation
            graph.add_edge("worker_setup", entry)
        
        # The conditional routing will override the direct edges for activation control
        # but the structural edges prevent orphaned nodes
        graph.add_conditional_edges(
            "worker_setup",
            route_to_worker,
            worker_entries
        )
        
        # Connect all worker exits to resolve_setup
        # The resolve_setup will handle checking if the selected worker has completed
        for worker_exit in worker_exits.values():
            graph.add_edge(worker_exit, "resolve_setup")
        
        # Connect resolve flow
        graph.add_edge("resolve_setup", resolve_entry)
        graph.add_edge(resolve_exit, END)
        
        return graph
    
    async def arun(self, messages, config=None):
        """Execute the case-based routing workflow."""
        compiled_graph = self.get_compiled_graph()
        
        initial_state = {
            "messages": messages,
            "agent_data": {}
        }
        
        # Execute the graph
        result = await compiled_graph.ainvoke(initial_state, config=config)
        
        # Extract the resolve worker's result using occurrence_id
        agent_data = result.get("agent_data", {})
        resolve_worker_occurrence_id = "resolve_worker"
        
        # Check decisions first (new location), then fallback to results (old location)
        decisions = agent_data.get("decisions", {})
        resolve_decision = decisions.get(resolve_worker_occurrence_id, {})
        
        if resolve_decision and "answer" in resolve_decision:
            response_text = resolve_decision["answer"]
        else:
            # Fallback to old results location
            results = agent_data.get("results", {})
            resolve_result = results.get(resolve_worker_occurrence_id, {})
            response_text = resolve_result.get("result_text", None)
        
        if response_text:
            
            # Add the final response to messages
            updated_messages = result["messages"] + [AIMessage(content=response_text)]
            
            return {
                "messages": updated_messages,
                "agent_data": agent_data
            }
        else:
            # Fallback if no resolve result
            fallback_message = "I processed your request but couldn't generate a proper response."
            updated_messages = result["messages"] + [AIMessage(content=fallback_message)]
            
            return {
                "messages": updated_messages,
                "agent_data": agent_data
            }
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about this case agent."""
        case_info = [
            {
                "id": case.id,
                "name": case.name,
                "description": case.description,
                "worker_name": self.workers[case.id].name,
                "worker_type": type(self.workers[case.id]).__name__
            }
            for case in self.cases
        ]
        
        return {
            "name": self.name,
            "type": "KGraphCaseAgent",
            "has_checkpointer": self.checkpointer is not None,
            "agent_type": "case_routing",
            "case_count": len(self.cases),
            "cases": case_info,
            "case_worker_name": self.case_worker.name,
            "resolve_worker_name": self.resolve_worker.name
        }
