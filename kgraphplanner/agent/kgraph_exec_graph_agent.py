from __future__ import annotations

import re
import logging
from typing import Dict, Any, List, Optional, Tuple

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.base import BaseCheckpointSaver

from kgraphplanner.agent.kgraph_base_agent import KGraphBaseAgent, AgentState
from kgraphplanner.worker.kgraph_worker import KGraphWorker
from kgraphplanner.graph.exec_graph import (
    GraphSpec, EdgeSpec, WorkerNodeSpec, StartNodeSpec, EndNodeSpec
)
from kgraphplanner.graph.graph_hop import merge_activation

logger = logging.getLogger(__name__)


def _safe_id(*parts: str) -> str:
    """Generate a safe node ID from parts."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", "__".join(parts))


class KGraphExecGraphAgent(KGraphBaseAgent):
    """
    Agent that executes a GraphSpec using a registry of KGraphWorker instances.
    
    Takes a declarative GraphSpec (produced by expanding a ProgramSpec) and builds
    a LangGraph execution graph where each WorkerNodeSpec maps to a worker's
    subgraph, connected by hop nodes that handle data binding and activation.
    """
    
    def __init__(
        self,
        *,
        name: str,
        graph_spec: GraphSpec,
        worker_registry: Dict[str, KGraphWorker],
        checkpointer: Optional[BaseCheckpointSaver] = None
    ):
        super().__init__(name=name, checkpointer=checkpointer)
        self.graph_spec = graph_spec
        self.worker_registry = worker_registry
        
        # Validate that all referenced workers exist in registry
        for node in graph_spec.nodes:
            if isinstance(node, WorkerNodeSpec):
                if node.worker_name not in worker_registry:
                    raise KeyError(
                        f"Worker '{node.worker_name}' for node '{node.id}' "
                        f"not found in registry. Available: {list(worker_registry.keys())}"
                    )
    
    def build_graph(self) -> StateGraph:
        """
        Build a LangGraph StateGraph from the GraphSpec.
        
        Structure:
            START → __entry__ → [hop → worker → hop → worker → ...] → END
            
        Each worker is spliced in via build_subgraph().
        Hop nodes between workers resolve bindings and set up activation data.
        """
        spec = self.graph_spec
        graph = StateGraph(AgentState)
        
        # Index nodes by id
        by_id = {node.id: node for node in spec.nodes}
        
        # --- Entry node: initialize agent_data and seed start results ---
        def entry_node(state: AgentState) -> AgentState:
            agent_data = dict(state.get("agent_data", {}))
            results = dict(agent_data.get("results", {}))
            
            # Seed start node's initial_data into results
            start_node = next(
                (n for n in spec.nodes if isinstance(n, StartNodeSpec)),
                None
            )
            if start_node and "start" not in results:
                results["start"] = start_node.initial_data.get("args", start_node.initial_data)
            
            return {
                "agent_data": {
                    **agent_data,
                    "results": results,
                    "errors": agent_data.get("errors", {}),
                    "activation": {},
                    "work": agent_data.get("work", {})
                }
            }
        
        graph.add_node("__entry__", entry_node)
        graph.add_edge(START, "__entry__")
        
        # --- Splice in worker subgraphs ---
        entry_for: Dict[str, str] = {}  # node_id -> subgraph entry label
        exit_for: Dict[str, str] = {}   # node_id -> subgraph exit label
        
        for node in spec.nodes:
            if not isinstance(node, WorkerNodeSpec):
                continue
            
            worker = self.worker_registry[node.worker_name]
            entry_label, exit_label = worker.build_subgraph(graph, node.id)
            entry_for[node.id] = entry_label
            exit_for[node.id] = exit_label
            logger.debug(f"Spliced worker '{node.worker_name}' as '{node.id}' "
                        f"(entry={entry_label}, exit={exit_label})")
        
        # --- Hop and gather node factories ---
        
        def _make_hop(edge: EdgeSpec, dest_node_id: str):
            """Create a hop node for a single incoming edge (no fan-in)."""
            dest_node = by_id.get(dest_node_id)
            dest_defaults = {}
            if isinstance(dest_node, WorkerNodeSpec):
                dest_defaults = dest_node.defaults or {}
            
            def hop(state: AgentState) -> AgentState:
                agent_data = dict(state.get("agent_data", {}))
                activation_map = dict(agent_data.get("activation", {}))
                results = agent_data.get("results", {})
                
                prev_act = activation_map.get(dest_node_id, {"prompt": "", "args": {}})
                act = merge_activation(prev_act, dest_defaults, edge, results)
                activation_map[dest_node_id] = act
                
                return {
                    "agent_data": {
                        **agent_data,
                        "activation": activation_map
                    }
                }
            
            return hop
        
        def _make_gather(edges: List[EdgeSpec], dest_node_id: str):
            """
            Fan-in gather node.  Runs once after ALL parallel source workers
            complete.  Iterates every incoming edge, accumulating bindings
            (append_list / merge_dict / concat_text) into a single activation.
            This avoids the shallow-merge data-loss bug where parallel hop
            nodes would each overwrite activation[dest_id].
            """
            dest_node = by_id.get(dest_node_id)
            dest_defaults = {}
            if isinstance(dest_node, WorkerNodeSpec):
                dest_defaults = dest_node.defaults or {}
            
            def gather(state: AgentState) -> AgentState:
                agent_data = dict(state.get("agent_data", {}))
                activation_map = dict(agent_data.get("activation", {}))
                results = agent_data.get("results", {})
                
                act = activation_map.get(dest_node_id, {"prompt": "", "args": {}})
                for edge in edges:
                    act = merge_activation(act, dest_defaults, edge, results)
                activation_map[dest_node_id] = act
                
                return {
                    "agent_data": {
                        **agent_data,
                        "activation": activation_map
                    }
                }
            
            return gather
        
        # --- Conditional routing support ---
        
        def _evaluate_condition(condition: str, result: Any) -> bool:
            """
            Evaluate a condition expression against a node's result.
            
            Supports:
              - "__default__": always False (used as fallback)
              - "true" / "always": always True
              - "has:<key>": True if result dict has key with truthy value
              - "eq:<key>:<value>": True if result[key] == value
              - Arbitrary Python expression evaluated with result in scope
            """
            if condition == "__default__":
                return False
            if condition in ("true", "always"):
                return True
            if condition.startswith("has:"):
                key = condition[4:]
                if isinstance(result, dict):
                    return bool(result.get(key))
                return False
            if condition.startswith("eq:"):
                parts = condition[3:].split(":", 1)
                if len(parts) == 2 and isinstance(result, dict):
                    return str(result.get(parts[0], "")) == parts[1]
                return False
            # Fallback: evaluate as expression with result in scope
            try:
                return bool(eval(condition, {"__builtins__": {}}, {"result": result}))
            except Exception:
                return False
        
        # --- Classify and group edges ---
        
        from collections import defaultdict
        
        start_edges: List[EdgeSpec] = []
        worker_edges: List[EdgeSpec] = []
        conditional_edges: List[EdgeSpec] = []
        
        for e in spec.edges:
            if e.source == "start" and e.destination in entry_for:
                start_edges.append(e)
            elif e.source in exit_for and e.destination in entry_for:
                if e.condition is not None:
                    conditional_edges.append(e)
                else:
                    worker_edges.append(e)
        
        # Group regular edges by destination to detect fan-in
        start_by_dest: Dict[str, List[EdgeSpec]] = defaultdict(list)
        for e in start_edges:
            start_by_dest[e.destination].append(e)
        
        worker_by_dest: Dict[str, List[EdgeSpec]] = defaultdict(list)
        for e in worker_edges:
            worker_by_dest[e.destination].append(e)
        
        # Group conditional edges by source
        cond_by_source: Dict[str, List[EdgeSpec]] = defaultdict(list)
        for e in conditional_edges:
            cond_by_source[e.source].append(e)
        
        # --- 1. Wire start → worker edges ---
        for dest_id, dest_edges in start_by_dest.items():
            if len(dest_edges) == 1:
                hop_id = _safe_id("__hop__", "start", "to", dest_id)
                graph.add_node(hop_id, _make_hop(dest_edges[0], dest_id))
                graph.add_edge("__entry__", hop_id)
                graph.add_edge(hop_id, entry_for[dest_id])
            else:
                gather_id = _safe_id("__gather__", "start", "to", dest_id)
                graph.add_node(gather_id, _make_gather(dest_edges, dest_id))
                graph.add_edge("__entry__", gather_id)
                graph.add_edge(gather_id, entry_for[dest_id])
        
        # --- 2. Auto-connect unreachable workers to entry ---
        if not start_edges:
            incoming = {n.id: 0 for n in spec.nodes}
            for e in spec.edges:
                incoming[e.destination] = incoming.get(e.destination, 0) + 1
            for node in spec.nodes:
                if not isinstance(node, WorkerNodeSpec):
                    continue
                if incoming.get(node.id, 0) == 0 and node.id in entry_for:
                    hop_id = _safe_id("__hop__", "__entry__", "to", node.id)
                    graph.add_node(hop_id, _make_hop(
                        EdgeSpec(source="__entry__", destination=node.id), node.id
                    ))
                    graph.add_edge("__entry__", hop_id)
                    graph.add_edge(hop_id, entry_for[node.id])
        
        # --- 3. Wire worker → worker edges (fan-in safe) ---
        for dest_id, dest_edges in worker_by_dest.items():
            source_exits = [exit_for[e.source] for e in dest_edges]
            
            if len(dest_edges) == 1:
                # Single incoming edge: simple hop
                e = dest_edges[0]
                hop_id = _safe_id("__hop__", e.source, "to", dest_id)
                graph.add_node(hop_id, _make_hop(e, dest_id))
                graph.add_edge(source_exits[0], hop_id)
                graph.add_edge(hop_id, entry_for[dest_id])
            else:
                # Fan-in: single gather node.
                # All source worker exits feed into it; LangGraph waits for
                # every predecessor before running the gather.  Workers
                # themselves still execute in parallel.
                gather_id = _safe_id("__gather__", dest_id)
                graph.add_node(gather_id, _make_gather(dest_edges, dest_id))
                for src_exit in source_exits:
                    graph.add_edge(src_exit, gather_id)
                graph.add_edge(gather_id, entry_for[dest_id])
                logger.info(f"Fan-in gather for '{dest_id}': "
                           f"{[e.source for e in dest_edges]} → {gather_id}")
        
        # --- 4. Wire conditional edges (source → router → branch hops) ---
        for source_id, cond_edges in cond_by_source.items():
            # Create a hop node for each branch destination
            route_map: Dict[str, str] = {}  # route_key → hop_node_id
            default_key = None
            
            for i, e in enumerate(cond_edges):
                dest_id = e.destination
                hop_id = _safe_id("__cond_hop__", source_id, "to", dest_id)
                graph.add_node(hop_id, _make_hop(e, dest_id))
                graph.add_edge(hop_id, entry_for[dest_id])
                
                route_key = f"branch_{i}" if e.condition != "__default__" else "__default__"
                route_map[route_key] = hop_id
                if e.condition == "__default__":
                    default_key = route_key
            
            # Build condition list for the router (excluding default)
            branch_conditions = []
            for i, e in enumerate(cond_edges):
                if e.condition != "__default__":
                    branch_conditions.append((f"branch_{i}", e.condition))
            
            # Create the router function
            src_id_captured = source_id
            conditions_captured = list(branch_conditions)
            default_captured = default_key
            
            def _make_router(src_id, conditions, default_route):
                def router(state: AgentState) -> str:
                    results = state.get("agent_data", {}).get("results", {})
                    result = results.get(src_id, {})
                    
                    for route_key, condition in conditions:
                        if _evaluate_condition(condition, result):
                            logger.info(f"Conditional routing '{src_id}': "
                                       f"condition '{condition}' matched → {route_key}")
                            return route_key
                    
                    if default_route:
                        logger.info(f"Conditional routing '{src_id}': "
                                   f"no condition matched → default")
                        return default_route
                    
                    # Fallback: first branch
                    fallback = conditions[0][0] if conditions else "__default__"
                    logger.warning(f"Conditional routing '{src_id}': "
                                  f"no match, no default → fallback {fallback}")
                    return fallback
                return router
            
            router_fn = _make_router(
                src_id_captured, conditions_captured, default_captured
            )
            
            graph.add_conditional_edges(
                exit_for[source_id],
                router_fn,
                route_map
            )
            
            logger.info(f"Conditional routing for '{source_id}': "
                       f"{len(cond_edges)} branches, "
                       f"default={'yes' if default_key else 'no'}")
        
        # --- 5. Connect exit nodes to END ---
        connected_to_end: set = set()
        
        for x in spec.exit_points:
            if x in exit_for and x not in connected_to_end:
                graph.add_edge(exit_for[x], END)
                connected_to_end.add(x)
        
        for e in spec.edges:
            if e.destination == "end" and e.source in exit_for and e.source not in connected_to_end:
                graph.add_edge(exit_for[e.source], END)
                connected_to_end.add(e.source)
        
        return graph
    
    async def arun(self, *, messages=None, config=None):
        """
        Execute the graph.
        
        Args:
            messages: Optional initial messages
            config: LangGraph config (must include configurable.thread_id)
            
        Returns:
            Final state with results from all workers in agent_data.results
        """
        compiled_graph = self.get_compiled_graph()
        
        initial_state = {
            "messages": messages or [],
            "agent_data": {},
            "work": {}
        }
        
        result = await compiled_graph.ainvoke(initial_state, config=config)
        return result
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about this execution graph agent."""
        base_info = super().get_agent_info()
        base_info.update({
            "agent_type": "exec_graph",
            "graph_id": self.graph_spec.graph_id,
            "node_count": len(self.graph_spec.nodes),
            "edge_count": len(self.graph_spec.edges),
            "worker_nodes": [
                n.id for n in self.graph_spec.nodes
                if isinstance(n, WorkerNodeSpec)
            ],
            "exit_points": self.graph_spec.exit_points
        })
        return base_info

