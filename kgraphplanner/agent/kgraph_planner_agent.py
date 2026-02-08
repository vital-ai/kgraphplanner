from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.base import BaseCheckpointSaver
from pydantic import ValidationError

from kgraphplanner.agent.kgraph_base_agent import KGraphBaseAgent, AgentState
from kgraphplanner.worker.kgraph_worker import KGraphWorker
from kgraphplanner.program.program import ProgramSpec
from kgraphplanner.program.program_expander import (
    validate_program_spec, expand_program_to_graph
)
from kgraphplanner.graph.exec_graph import (
    GraphSpec, EdgeSpec, WorkerNodeSpec, StartNodeSpec, EndNodeSpec,
    validate_graph_spec
)
from kgraphplanner.graph.graph_hop import merge_activation

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = """You are a planning compiler. Return a compact ProgramSpec (JSON) to be expanded into a GraphSpec and executed by workers.

Rules:
1) Always include program_id, static_nodes (with "start" and "end"), static_edges, and exit_nodes: ["end"].
2) Use a static node "start" with initial data under start.defaults.args (e.g., {"topic": "AI"}).
3) For simple linear or branching workflows, add worker nodes directly to static_nodes with worker name matching an available worker, and connect them via static_edges.
4) For iteration over collections (e.g., multiple companies), use TemplateSpec.for_each with item_var and idx_var to generate nodes dynamically.
5) Prefer the sugar fields for parallel patterns:
   - fan_out: one source -> multiple destinations (each branch has its own bindings/prompt).
   - fan_in: many sources -> one destination with a reducer (append_list/merge_dict/concat_text).
   Use loop_edges only for non-fan edges (e.g., start -> first step in the loop).
6) Do NOT enumerate parallel edges in loop_edges; use fan_out/fan_in instead.
7) Workers referenced MUST exist in the available worker registry.
8) Each static_edge needs source and destination. Optionally include prompt (task instruction for the destination worker).
9) CRITICAL — Bindings are REQUIRED on every worker→worker edge. Without bindings, downstream workers receive NO data.
   Each worker stores its output at $.result_text. Rules:
   a) Every static_edge between workers MUST have bindings, e.g.: "bindings": {"input": [{"from_node": "source_id", "path": "$.result_text"}]}
   b) Every loop_edge MUST have bindings. For start→worker edges in a for_each loop, bind the item: "bindings": {"company": [{"from_node_tpl": "start", "path": "$.companies[{idx}]"}]}
   c) Every fan_out branch MUST have bindings to forward the source's output: "bindings": {"input": [{"from_node_tpl": "research_{idx}", "path": "$.result_text"}]}
   d) fan_in uses param+reduce instead of per-edge bindings (this is correct as-is).
10) Keep it succinct; never emit an already-expanded graph. No duplicates.

Return ONLY a valid ProgramSpec JSON object."""

PLANNER_FEW_SHOT_EXAMPLE = """{
  "program_id": "company_analysis",
  "static_nodes": [
    {"id":"start","worker":"start","defaults":{"args":{"companies":["OpenAI","Anthropic","Cohere"]}}},
    {"id":"aggregate","worker":"aggregator"},
    {"id":"end","worker":"end"}
  ],
  "static_edges": [{"source":"aggregate","destination":"end"}],
  "templates": [
    {
      "name": "per_company",
      "for_each": {"source_from":"start_args","source_path":"$.companies","item_var":"company","idx_var":"idx"},
      "loop_nodes": [
        {"id_tpl":"research_{idx}","worker":"research_worker",
         "defaults_tpl":{"prompt":"Research {company}","args":{"topic":"{company}"}}},
        {"id_tpl":"analysis_a_{idx}","worker":"analyst_a"},
        {"id_tpl":"analysis_b_{idx}","worker":"analyst_b"}
      ],
      "loop_edges": [
        {"source_tpl":"start","destination_tpl":"research_{idx}",
         "prompt_tpl":"Research the company {company}",
         "bindings":{"company":[{"from_node_tpl":"start","path":"$.companies[{idx}]","reduce":"overwrite"}]}}
      ],
      "fan_out": [
        {"source_tpl":"research_{idx}",
         "branches":[
           {"destination_tpl":"analysis_a_{idx}",
            "prompt_tpl":"Analyze track A for {company}",
            "bindings":{"input":[{"from_node_tpl":"research_{idx}","path":"$"}]}},
           {"destination_tpl":"analysis_b_{idx}",
            "prompt_tpl":"Analyze track B for {company}",
            "bindings":{"input":[{"from_node_tpl":"research_{idx}","path":"$"}]}}
         ]}
      ],
      "fan_in": [
        {"sources_tpl":["analysis_a_{idx}","analysis_b_{idx}"],
         "destination_tpl":"aggregate",
         "param":"analyses",
         "reduce":"append_list",
         "prompt_tpl":"Include analysis for {company}"}
      ]
    }
  ],
  "exit_nodes": ["end"],
  "max_parallel": 3
}"""

PLANNER_SIMPLE_EXAMPLE = """{
  "program_id": "research_and_summarize",
  "static_nodes": [
    {"id":"start","worker":"start","defaults":{"args":{"topic":"artificial intelligence"}}},
    {"id":"research_node","worker":"researcher","defaults":{"prompt":"Research the given topic thoroughly"}},
    {"id":"summary_node","worker":"summarizer","defaults":{"prompt":"Summarize the research findings concisely"}},
    {"id":"end","worker":"end"}
  ],
  "static_edges": [
    {"source":"start","destination":"research_node","prompt":"Research the topic",
     "bindings":{"topic":[{"from_node":"start","path":"$.topic"}]}},
    {"source":"research_node","destination":"summary_node","prompt":"Summarize the research",
     "bindings":{"input":[{"from_node":"research_node","path":"$.result_text"}]}},
    {"source":"summary_node","destination":"end"}
  ],
  "exit_nodes": ["end"],
  "max_parallel": 3
}"""


class KGraphPlannerAgent(KGraphBaseAgent):
    """
    Agent that plans and executes a dynamic workflow from a natural language request.
    
    Pipeline:
        User Request -> [LLM Planner] -> ProgramSpec -> [Expander] -> GraphSpec -> [Executor] -> Results
    
    The planner uses an LLM with structured output to generate a ProgramSpec,
    which is then expanded into a GraphSpec and executed using KGraphExecGraphAgent's
    graph-building logic.
    """
    
    def __init__(
        self,
        *,
        name: str,
        planner_llm: ChatOpenAI,
        worker_registry: Dict[str, KGraphWorker],
        execution_llm: Optional[ChatOpenAI] = None,
        system_prompt: Optional[str] = None,
        checkpointer: Optional[BaseCheckpointSaver] = None
    ):
        super().__init__(name=name, checkpointer=checkpointer)
        self.planner_llm = planner_llm
        self.worker_registry = worker_registry
        self.execution_llm = execution_llm or planner_llm
        self.system_prompt = system_prompt or PLANNER_SYSTEM_PROMPT
    
    async def plan(self, messages: List[BaseMessage], max_retries: int = 2) -> ProgramSpec:
        """
        Use the planner LLM to generate a ProgramSpec from user messages.
        
        Uses few-shot prompting with AIMessage examples to teach the LLM
        the exact JSON shape for both simple (static-only) and complex
        (template with fan-out/fan-in) workflows.
        
        Retries up to max_retries times on schema or validation failures,
        feeding the error back to the LLM as context.
        
        Args:
            messages: Conversation messages (typically ending with a HumanMessage)
            max_retries: Number of retry attempts on validation failure (default 2)
            
        Returns:
            Validated ProgramSpec
            
        Raises:
            ValueError: If all attempts fail validation
        """
        model = self.planner_llm.with_structured_output(
            ProgramSpec, method="function_calling"
        )
        
        available_workers = list(self.worker_registry.keys())
        
        sys_msg = SystemMessage(content=(
            self.system_prompt +
            f"\n\nAvailable workers: {available_workers}"
        ))
        
        # Few-shot examples as AIMessages (teaches the LLM the exact JSON shape)
        simple_prompt = HumanMessage(content="Research a topic then summarize the findings.")
        simple_example = AIMessage(content=PLANNER_SIMPLE_EXAMPLE)
        
        complex_prompt = HumanMessage(content="Analyze multiple companies with two analysis tracks each, then aggregate.")
        complex_example = AIMessage(content=PLANNER_FEW_SHOT_EXAMPLE)
        
        base_messages = [
            sys_msg,
            simple_prompt, simple_example,
            complex_prompt, complex_example,
            *messages
        ]
        
        last_error = None
        for attempt in range(1 + max_retries):
            all_messages = list(base_messages)
            
            # On retry, append the previous error so the LLM can self-correct
            if last_error and attempt > 0:
                logger.info(f"Retry {attempt}/{max_retries}: feeding error back to LLM")
                all_messages.append(HumanMessage(
                    content=f"Your previous ProgramSpec was invalid: {last_error}\n"
                            f"Please fix the issue and try again."
                ))
            
            logger.info(f"Generating ProgramSpec (attempt {attempt + 1}/{1 + max_retries})...")
            try:
                program = await model.ainvoke(all_messages)
            except ValidationError as e:
                last_error = str(e)
                logger.warning(f"Attempt {attempt + 1} schema validation failed: {last_error}")
                continue
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Attempt {attempt + 1} LLM error: {last_error}")
                continue
            
            # Validate
            is_valid, validation_messages = validate_program_spec(program)
            if not is_valid:
                last_error = "; ".join(validation_messages)
                logger.warning(f"Attempt {attempt + 1} program validation failed: {last_error}")
                continue
            
            for msg in validation_messages:
                if msg.startswith("WARNING"):
                    logger.warning(msg)
            
            if attempt > 0:
                logger.info(f"ProgramSpec succeeded on retry {attempt}")
            
            logger.info(f"ProgramSpec generated: {program.program_id} "
                        f"({len(program.static_nodes)} static nodes, "
                        f"{len(program.templates)} templates)")
            return program
        
        raise ValueError(
            f"LLM failed to generate valid ProgramSpec after {1 + max_retries} attempts. "
            f"Last error: {last_error}"
        )
    
    def expand(self, program: ProgramSpec) -> GraphSpec:
        """
        Expand a ProgramSpec into a GraphSpec.
        
        Args:
            program: Validated ProgramSpec
            
        Returns:
            Expanded GraphSpec
            
        Raises:
            ValueError: If the expanded graph fails validation
        """
        graph_spec = expand_program_to_graph(program)
        
        validation_result = validate_graph_spec(graph_spec)
        if validation_result.has_errors():
            errors = [e.message for e in validation_result.errors]
            raise ValueError(f"Expanded graph has errors: {'; '.join(errors)}")
        
        if validation_result.has_warnings():
            for w in validation_result.warnings:
                logger.warning(f"Graph warning: {w.message}")
        
        logger.info(f"GraphSpec expanded: {len(graph_spec.nodes)} nodes, "
                    f"{len(graph_spec.edges)} edges")
        return graph_spec
    
    def _build_exec_graph(self, graph_spec: GraphSpec) -> StateGraph:
        """
        Build a LangGraph StateGraph from GraphSpec using the worker registry.
        Delegates to KGraphExecGraphAgent.build_graph() which handles edge
        grouping, gather nodes for fan-in, conditional routing, and parallel
        execution correctly.
        """
        from kgraphplanner.agent.kgraph_exec_graph_agent import KGraphExecGraphAgent
        
        exec_agent = KGraphExecGraphAgent(
            name=f"{self.name}_exec",
            graph_spec=graph_spec,
            worker_registry=self.worker_registry
        )
        return exec_agent.build_graph()
    
    def build_graph(self) -> StateGraph:
        """
        Not used directly — the planner agent builds the graph dynamically
        in arun() after generating the ProgramSpec. This returns a minimal
        placeholder graph.
        """
        graph = StateGraph(AgentState)
        
        def passthrough(state: AgentState) -> AgentState:
            return state
        
        graph.add_node("passthrough", passthrough)
        graph.add_edge(START, "passthrough")
        graph.add_edge("passthrough", END)
        return graph
    
    async def arun(self, *, messages: List[BaseMessage], config=None) -> Dict[str, Any]:
        """
        Plan, expand, and execute a workflow from user messages.
        
        Args:
            messages: User messages (typically [HumanMessage("...")])
            config: LangGraph config (must include configurable.thread_id)
            
        Returns:
            Dict with:
                - agent_data.results: Results from all workers
                - agent_data.errors: Any errors
                - program_spec: The generated ProgramSpec
                - graph_spec: The expanded GraphSpec
        """
        # Step 1: Plan
        logger.info("Step 1: Planning...")
        try:
            program = await self.plan(messages)
        except (ValidationError, ValueError) as e:
            logger.error(f"Planning failed: {e}")
            return {
                "messages": messages,
                "agent_data": {"errors": {"planner": str(e)}, "results": {}},
                "program_spec": None,
                "graph_spec": None
            }
        
        # Step 2: Expand
        logger.info("Step 2: Expanding...")
        try:
            graph_spec = self.expand(program)
        except ValueError as e:
            logger.error(f"Expansion failed: {e}")
            return {
                "messages": messages,
                "agent_data": {"errors": {"expander": str(e)}, "results": {}},
                "program_spec": program.model_dump(),
                "graph_spec": None
            }
        
        # Step 3: Execute
        logger.info("Step 3: Executing...")
        exec_graph = self._build_exec_graph(graph_spec)
        compiled = exec_graph.compile(checkpointer=self.checkpointer)
        
        initial_state = {
            "messages": messages,
            "agent_data": {},
            "work": {}
        }
        
        result = await compiled.ainvoke(initial_state, config=config)
        
        # Attach specs to result for inspection
        result["program_spec"] = program.model_dump()
        result["graph_spec"] = graph_spec.model_dump()
        
        agent_data = result.get("agent_data", {})
        results = agent_data.get("results", {})
        errors = agent_data.get("errors", {})
        logger.info(f"Execution complete: {len(results)} results, {len(errors)} errors")
        
        return result
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about this planner agent."""
        base_info = super().get_agent_info()
        base_info.update({
            "agent_type": "planner",
            "planner_model": self.planner_llm.model_name if hasattr(self.planner_llm, 'model_name') else str(self.planner_llm),
            "available_workers": list(self.worker_registry.keys())
        })
        return base_info
