from __future__ import annotations

import asyncio
import contextlib
import copy
import re
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.base import BaseCheckpointSaver

from kgraphplanner.agent.kgraph_base_agent import (
    KGraphBaseAgent, AgentState, replace_value, append_values, merge_agent_data,
)
from kgraphplanner.agent.node_runtime import (
    DEFAULT_DEADLINE_GRACE, NodePolicyGraph, run_context,
)
from kgraphplanner.worker.llm_usage import model_name, usage_state_update
from kgraphplanner.worker.worker_errors import error_type_of
from kgraphplanner.worker.kgraph_worker import KGraphWorker
from kgraphplanner.graph.exec_graph import (
    GraphSpec, EdgeSpec, WorkerNodeSpec, StartNodeSpec,
    validate_graph_spec,
)
from kgraphplanner.graph.conditions import parse_condition, evaluate_condition
from kgraphplanner.graph.exec_errors import (
    GraphSpecError, ExecGraphRoutingError, ExecGraphLoopExhausted,
)
from kgraphplanner.graph.graph_hop import merge_activation

logger = logging.getLogger(__name__)

DEFAULT_RECURSION_LIMIT = 200
# run()'s backstop fires this long after deadline + grace, so node-level
# budgets (which end at deadline + grace at the latest) always fire first and
# the error is recorded, rather than the whole run being cancelled.
BACKSTOP_MARGIN = 0.25


def _safe_id(*parts: str) -> str:
    """Generate a safe node ID from parts."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", "__".join(parts))


def edge_key(source: str, destination: str) -> str:
    """Key for ``agent_data.traversals`` and the ``traversals`` condition name."""
    return f"{source}->{destination}"


def is_llm_worker(worker: KGraphWorker) -> bool:
    """True for workers that call an LLM (and can take a "final turn" prompt)."""
    return bool(getattr(worker, "uses_llm", True)) and getattr(worker, "llm", None) is not None


@dataclass
class RunResult:
    """What ``KGraphExecGraphAgent.run`` returns.

    Also attached as ``exc.run_result`` to any exception ``run()`` raises,
    holding whatever the run produced before it failed.
    """
    results: Dict[str, Any]
    errors: Dict[str, Any]
    usage: Dict[str, Any]
    usage_log: Dict[str, Any]
    run: Dict[str, Any]
    trace: List[str]
    state: Dict[str, Any] = field(repr=False, default_factory=dict)


class KGraphExecGraphAgent(KGraphBaseAgent):
    """
    Agent that executes a GraphSpec using a registry of KGraphWorker instances.

    Takes a declarative GraphSpec (produced by expanding a ProgramSpec) and builds
    a LangGraph execution graph where each WorkerNodeSpec maps to a worker's
    subgraph, connected by hop nodes that handle data binding and activation.

    The spec is validated in the constructor; an invalid spec raises
    ``GraphSpecError`` listing every problem.

    Args:
        allow_expressions: allow Python-expression conditions. Turn off for
            specs that come from an LLM (``eval`` is not a sandbox).
        inject_last_human_as: when ``arun`` gets messages, copy the last
            HumanMessage into ``results["start"][<this key>]`` so bindings can
            read it. ``None`` turns this off. ``run()`` never injects.
    """

    def __init__(
        self,
        *,
        name: str,
        graph_spec: GraphSpec,
        worker_registry: Dict[str, KGraphWorker],
        checkpointer: Optional[BaseCheckpointSaver] = None,
        allow_expressions: bool = True,
        inject_last_human_as: Optional[str] = "input",
    ):
        super().__init__(name=name, checkpointer=checkpointer)
        self.graph_spec = graph_spec
        self.worker_registry = worker_registry
        self.allow_expressions = allow_expressions
        self.inject_last_human_as = inject_last_human_as
        self._compiled_graph_no_checkpointer = None

        # Validate that all referenced workers exist in registry
        for node in graph_spec.nodes:
            if isinstance(node, WorkerNodeSpec):
                if node.worker_name not in worker_registry:
                    raise KeyError(
                        f"Worker '{node.worker_name}' for node '{node.id}' "
                        f"not found in registry. Available: {list(worker_registry.keys())}"
                    )
        self._validate()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _worker_ids(self) -> set:
        return {n.id for n in self.graph_spec.nodes if isinstance(n, WorkerNodeSpec)}

    def _classify_edge(self, e: EdgeSpec, worker_ids: set) -> Optional[str]:
        """How build_graph wires an edge, or None if it would be dropped."""
        if e.source == "start":
            return "start" if e.destination in worker_ids and e.condition is None else None
        if e.source not in worker_ids:
            return None
        if e.condition is not None:
            return "conditional" if (e.destination in worker_ids or e.destination == "end") else None
        if e.destination in worker_ids:
            return "worker"
        if e.destination == "end":
            return "end"
        return None

    def resolve_on_exhausted(self, e: EdgeSpec) -> str:
        """``on_exhausted``, with None resolved by the source worker's type."""
        if e.on_exhausted is not None:
            return e.on_exhausted
        node = self.graph_spec.get_node_by_id(e.source)
        worker = self.worker_registry.get(node.worker_name) if isinstance(node, WorkerNodeSpec) else None
        return "source_final_turn" if worker is not None and is_llm_worker(worker) else "error"

    def _validate(self) -> None:
        spec = self.graph_spec
        errors: List[str] = []
        warnings: List[str] = []

        for key, worker in self.worker_registry.items():
            if getattr(worker, "name", key) != key:
                errors.append(
                    f"Registry key '{key}' does not match the worker's name "
                    f"'{worker.name}'; stream events and logs would report the wrong worker")

        result = validate_graph_spec(spec, allow_expressions=self.allow_expressions)
        errors.extend(e.message for e in result.errors)
        warnings.extend(w.message for w in result.warnings)

        worker_ids = self._worker_ids()
        incoming_from_workers = defaultdict(list)
        for e in spec.edges:
            kind = self._classify_edge(e, worker_ids)
            if kind is None:
                if e.source == "start" and e.condition is not None:
                    why = "conditions on start edges are not supported"
                elif e.source not in worker_ids and e.source != "start":
                    why = f"source '{e.source}' is not a worker node"
                else:
                    why = f"destination '{e.destination}' is not a worker node or 'end'"
                errors.append(f"Edge '{e.source}' -> '{e.destination}' would be dropped: {why}")
            elif kind == "worker":
                incoming_from_workers[e.destination].append(e.source)

        for e in spec.edges:
            if self._classify_edge(e, worker_ids) == "start" and e.destination in incoming_from_workers:
                warnings.append(
                    f"Start edge to '{e.destination}' will be pruned: it is also reached "
                    f"from {incoming_from_workers[e.destination]}")
            if e.condition is not None and e.max_traversals is not None \
                    and e.on_exhausted == "source_final_turn":
                node = spec.get_node_by_id(e.source)
                worker = self.worker_registry.get(node.worker_name) if isinstance(node, WorkerNodeSpec) else None
                if worker is None or not is_llm_worker(worker):
                    errors.append(
                        f"Edge '{e.source}' -> '{e.destination}' has "
                        f"on_exhausted='source_final_turn', but '{e.source}' is not an LLM "
                        f"worker; use 'default' or 'error'")

        ignored = {"tools": [], "required_inputs": [], "max_iterations": 6}
        for node in spec.nodes:
            if isinstance(node, WorkerNodeSpec):
                set_fields = [f for f, default in ignored.items() if getattr(node, f) != default]
                if set_fields:
                    warnings.append(
                        f"Node '{node.id}' sets {set_fields}, which KGraphExecGraphAgent "
                        f"ignores; configure them on the worker instead")

        for w in warnings:
            logger.warning(f"GraphSpec '{spec.graph_id}': {w}")
        if errors:
            raise GraphSpecError(errors)

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

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
        start_node = next((n for n in spec.nodes if isinstance(n, StartNodeSpec)), None)
        inject_key = self.inject_last_human_as

        # --- Entry node: reset per-run state and seed start results ---
        def entry_node(state: AgentState) -> AgentState:
            agent_data = state.get("agent_data") or {}
            results = agent_data.get("results") or {}
            run_input = agent_data.get("run_input") or {}

            # Never write into dicts held by state or the caller: copy first.
            start = copy.deepcopy(results["start"]) if "start" in results else None
            if start is None and start_node is not None:
                start = copy.deepcopy(start_node.initial_data.get("args", start_node.initial_data))
            start_data = run_input.get("start_data")
            if start_data is not None:
                start = {**(start if isinstance(start, dict) else {}), **copy.deepcopy(start_data)}

            # Copy the last human message into start[inject_key] so that
            # bindings referencing start.input get the actual user input
            if inject_key:
                last_human = None
                for msg in reversed(state.get("messages") or []):
                    if msg.__class__.__name__ == "HumanMessage":
                        last_human = msg
                        break
                if last_human is not None:
                    start = start if isinstance(start, dict) else {}
                    start[inject_key] = last_human.content

            now = time.time()
            deadline = run_input.get("deadline")
            if deadline is None and spec.timeout_seconds:
                deadline = now + spec.timeout_seconds
            grace = run_input.get("grace", DEFAULT_DEADLINE_GRACE)

            # Everything below is per run. On a reused checkpointed thread,
            # results from an earlier turn would otherwise be visible to
            # conditions and functions for nodes that haven't run this turn.
            # Only the conversation (messages) carries over.
            update: Dict[str, Any] = {
                "results": replace_value({"start": start} if start is not None else {}),
                "decisions": replace_value({}),
                "activation": replace_value({}),
                "traversals": replace_value({}),
                "errors": replace_value({}),
                "trace": replace_value([]),
                "failed_nodes": replace_value({}),
                "usage_log": replace_value({}),
                "usage_total": replace_value({}),
                "run": replace_value({"started_at": now, "deadline": deadline, "grace": grace}),
                "run_input": replace_value(None),
            }
            # Worker scratch slots (e.g. the tool worker's history) too.
            stale_work = {slot: {} for slot in (state.get("work") or {})}
            return {"agent_data": update, "work": stale_work}

        graph.add_node("__entry__", entry_node)
        graph.add_edge(START, "__entry__")

        # --- Splice in worker subgraphs ---
        entry_for: Dict[str, str] = {}  # node_id -> subgraph entry label
        exit_for: Dict[str, str] = {}   # node_id -> subgraph exit label

        for node in spec.nodes:
            if not isinstance(node, WorkerNodeSpec):
                continue

            worker = self.worker_registry[node.worker_name]
            # on_error="record": a failed node's error result flows on to a
            # pass-through done node, and the graph's routing for this node
            # hangs off it (see node_runtime).
            done_id = _safe_id("__done__", node.id) if node.on_error == "record" else None
            llm_worker = is_llm_worker(worker)
            policy_graph = NodePolicyGraph(
                graph, node, done_id, uses_llm=llm_worker,
                model=(model_name(worker.llm) or "unknown") if llm_worker else None)
            entry_label, exit_label = worker.build_subgraph(policy_graph, node.id)
            if done_id is not None:
                graph.add_node(done_id, lambda state: {})
                graph.add_edge(exit_label, done_id)
                exit_label = done_id
            entry_for[node.id] = entry_label
            exit_for[node.id] = exit_label
            logger.debug(f"Spliced worker '{node.worker_name}' as '{node.id}' "
                        f"(entry={entry_label}, exit={exit_label})")

        # --- Hop and gather node factories ---
        # Hops return only the keys they change. Returning the whole
        # agent_data snapshot would overwrite concurrent writes from
        # parallel branches.

        def _dest_defaults(dest_node_id: str) -> Tuple[Dict[str, Any], bool]:
            """(defaults, reset) for a destination node. A node-level
            system_directive becomes a default arg, so a binding can still
            override it per run."""
            dest_node = by_id.get(dest_node_id)
            if not isinstance(dest_node, WorkerNodeSpec):
                return {}, False
            defaults = dest_node.defaults or {}
            if dest_node.system_directive:
                defaults = {**defaults, "args": {
                    "system_directive": dest_node.system_directive,
                    **(defaults.get("args") or {}),
                }}
            return defaults, dest_node.reentry == "reset"

        def _make_hop(edge: EdgeSpec, dest_node_id: str, count_traversal: bool = False):
            """Create a hop node for a single incoming edge (no fan-in)."""
            dest_defaults, reset = _dest_defaults(dest_node_id)
            ek = edge_key(edge.source, dest_node_id)

            def hop(state: AgentState) -> AgentState:
                agent_data = state.get("agent_data") or {}
                results = agent_data.get("results") or {}

                prev_act = {} if reset else (agent_data.get("activation") or {}).get(
                    dest_node_id, {"prompt": "", "args": {}})
                act = merge_activation(prev_act, dest_defaults, edge, results)

                update: Dict[str, Any] = {
                    "activation": {dest_node_id: act},
                    "trace": append_values([dest_node_id]),
                    "failed_nodes": {dest_node_id: False},  # a new activation
                }
                if count_traversal:
                    count = (agent_data.get("traversals") or {}).get(ek, 0) + 1
                    update["traversals"] = {ek: count}
                return {"agent_data": update}

            return hop

        def _make_gather(edges: List[EdgeSpec], dest_node_id: str):
            """
            Fan-in gather node.  Runs once after ALL parallel source workers
            complete.  Iterates every incoming edge, accumulating bindings
            (append_list / merge_dict / concat_text) into a single activation.
            This avoids the shallow-merge data-loss bug where parallel hop
            nodes would each overwrite activation[dest_id].
            """
            dest_defaults, reset = _dest_defaults(dest_node_id)

            def gather(state: AgentState) -> AgentState:
                agent_data = state.get("agent_data") or {}
                results = agent_data.get("results") or {}

                act = {"prompt": "", "args": {}} if reset else (agent_data.get("activation") or {}).get(
                    dest_node_id, {"prompt": "", "args": {}})
                for edge in edges:
                    act = merge_activation(act, dest_defaults, edge, results)

                return {"agent_data": {
                    "activation": {dest_node_id: act},
                    "trace": append_values([dest_node_id]),
                    "failed_nodes": {dest_node_id: False},  # a new activation
                }}

            return gather

        # --- Classify and group edges ---
        worker_ids = set(entry_for)
        start_edges: List[EdgeSpec] = []
        worker_edges: List[EdgeSpec] = []
        conditional_edges: List[EdgeSpec] = []
        end_edges: List[EdgeSpec] = []

        for e in spec.edges:
            kind = self._classify_edge(e, worker_ids)
            if kind == "start":
                start_edges.append(e)
            elif kind == "worker":
                worker_edges.append(e)
            elif kind == "conditional":
                conditional_edges.append(e)
            elif kind == "end":
                end_edges.append(e)

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
        # Prune redundant start edges: if a destination also has incoming
        # worker edges, the start edge would fire before the upstream worker
        # completes, bypassing the required data flow.
        for dest_id, dest_edges in start_by_dest.items():
            if dest_id in worker_by_dest:
                logger.info(
                    f"Pruning redundant start edge(s) to '{dest_id}' — "
                    f"already reached via worker edge(s) from "
                    f"{[e.source for e in worker_by_dest[dest_id]]}"
                )
                continue
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
                # All source worker exits feed into it.  For DAG fan-in,
                # LangGraph waits for every predecessor before running the
                # gather.  For cycles, LangGraph only waits for predecessors
                # activated in the current super-step (no deadlock).
                gather_id = _safe_id("__gather__", dest_id)
                graph.add_node(gather_id, _make_gather(dest_edges, dest_id))
                for src_exit in source_exits:
                    graph.add_edge(src_exit, gather_id)
                graph.add_edge(gather_id, entry_for[dest_id])
                logger.info(f"Fan-in gather for '{dest_id}': "
                           f"{[e.source for e in dest_edges]} → {gather_id}")

        # --- 4. Wire conditional edges (source → router → branch hops / END) ---
        used_ids: set = set()
        for source_id, cond_edges in cond_by_source.items():
            route_map: Dict[str, str] = {}  # route_key → hop node id or END
            branches = []  # (route_key, parsed condition, edge)
            default_key: Optional[str] = None
            loop_guard_key: Optional[str] = None

            for i, e in enumerate(cond_edges):
                parsed = parse_condition(e.condition, source=source_id,
                                         allow_expressions=self.allow_expressions)
                route_key = "__default__" if parsed.is_default else f"branch_{i}"
                if e.destination == "end":
                    route_map[route_key] = END
                else:
                    hop_id = _safe_id("__cond_hop__", source_id, "to", e.destination)
                    if hop_id in used_ids:  # two edges to the same destination
                        hop_id = _safe_id("__cond_hop__", source_id, str(i), "to", e.destination)
                    used_ids.add(hop_id)
                    graph.add_node(hop_id, _make_hop(e, e.destination, count_traversal=True))
                    graph.add_edge(hop_id, entry_for[e.destination])
                    route_map[route_key] = hop_id
                if parsed.is_default:
                    default_key = route_key
                else:
                    branches.append((route_key, parsed, e))

                # An exhausted edge whose policy is source_final_turn routes
                # to a loop-guard hop that re-activates the SOURCE so the
                # chat worker gets one final turn to compose its answer.
                if (e.max_traversals is not None and loop_guard_key is None
                        and self.resolve_on_exhausted(e) == "source_final_turn"):
                    lg_hop_id = _safe_id("__loop_guard__", source_id)
                    src_node = by_id.get(source_id)
                    src_defaults = (src_node.defaults or {}) if isinstance(src_node, WorkerNodeSpec) else {}
                    graph.add_node(lg_hop_id, self._make_loop_guard_hop(source_id, src_defaults))
                    graph.add_edge(lg_hop_id, entry_for[source_id])
                    loop_guard_key = "__loop_guard__"
                    route_map[loop_guard_key] = lg_hop_id
                    logger.info(f"Loop guard branch added for '{source_id}' "
                               f"(max_traversals={e.max_traversals})")

            router_fn = self._make_router(source_id, branches, default_key, loop_guard_key)
            graph.add_conditional_edges(exit_for[source_id], router_fn, route_map)

            logger.info(f"Conditional routing for '{source_id}': "
                       f"{len(cond_edges)} branches, "
                       f"default={'yes' if default_key else 'no'}, "
                       f"loop_guard={'yes' if loop_guard_key else 'no'}")

        # --- 5. Connect exit nodes to END ---
        # Sources with conditional edges reach END through their router only;
        # a static END edge would fire alongside whichever branch matched.
        connected_to_end: set = set(cond_by_source)

        for x in spec.exit_points:
            if x in exit_for and x not in connected_to_end:
                graph.add_edge(exit_for[x], END)
                connected_to_end.add(x)

        for e in end_edges:
            if e.source not in connected_to_end:
                graph.add_edge(exit_for[e.source], END)
                connected_to_end.add(e.source)

        return graph

    @staticmethod
    def _make_loop_guard_hop(src_id: str, src_defs: Dict[str, Any]):
        def loop_guard_hop(state: AgentState) -> AgentState:
            results = (state.get("agent_data") or {}).get("results") or {}

            # Collect whatever tool results exist
            tool_results_summary = ""
            for node_id, res in results.items():
                if node_id != src_id and isinstance(res, dict):
                    rt = res.get("result_text", "")
                    if rt:
                        tool_results_summary += f"\n{node_id}: {rt}"

            act = {
                "prompt": (
                    "IMPORTANT: You have reached the maximum number of "
                    "tool call cycles. You MUST NOT request any more "
                    "tool calls. Compose your final answer NOW using "
                    "action=\"respond\" based on the tool results you "
                    "have already gathered."
                ),
                "args": {
                    **src_defs.get("args", {}),
                    "tool_results_so_far": tool_results_summary,
                },
            }
            return {"agent_data": {
                "activation": {src_id: act},
                "trace": append_values([src_id]),
                "failed_nodes": {src_id: False},
            }}
        return loop_guard_hop

    def _make_router(self, src_id: str, branches, default_route: Optional[str],
                     loop_guard_route: Optional[str]):
        """Router for one source. Reads all run state from the graph state, so
        one compiled graph is safe to share across concurrent runs."""
        policies = {id(e): self.resolve_on_exhausted(e) for _, _, e in branches}

        def router(state: AgentState) -> str:
            agent_data = state.get("agent_data") or {}
            results = agent_data.get("results") or {}
            result = results.get(src_id, {})
            traversals = agent_data.get("traversals") or {}
            context = {
                "result": result,
                "results": results,
                "traversals": traversals,
                "run": run_context(state),
            }

            for route_key, cond, e in branches:
                if not evaluate_condition(cond, source=src_id, context=context):
                    continue
                if e.max_traversals is not None:
                    taken = traversals.get(edge_key(src_id, e.destination), 0)
                    if taken >= e.max_traversals:
                        policy = policies[id(e)]
                        logger.warning(
                            f"⚠️ Loop guard: edge '{src_id}' → '{e.destination}' "
                            f"hit max_traversals ({e.max_traversals}); on_exhausted={policy}")
                        if policy == "source_final_turn" and loop_guard_route:
                            return loop_guard_route
                        if policy == "default" and default_route:
                            return default_route
                        raise ExecGraphLoopExhausted(src_id, e.destination, e.max_traversals)
                logger.info(f"Conditional routing '{src_id}': "
                           f"condition '{cond.text}' matched → {route_key}")
                return route_key

            if default_route:
                logger.info(f"Conditional routing '{src_id}': "
                           f"no condition matched → default")
                return default_route
            raise ExecGraphRoutingError(src_id, result)

        return router

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _get_compiled_graph_no_checkpointer(self):
        if self._compiled_graph_no_checkpointer is None:
            self._compiled_graph_no_checkpointer = self.build_graph().compile(checkpointer=None)
        return self._compiled_graph_no_checkpointer

    async def arun(self, *, messages=None, config=None, start_data: Optional[Dict[str, Any]] = None,
                   deadline: Optional[float] = None, grace: Optional[float] = None):
        """
        Execute the graph.
        
        Args:
            messages: Optional initial messages
            config: LangGraph config. Needs ``configurable.thread_id`` only when
                the agent has a checkpointer. Not modified.
            start_data: Optional dict merged over the start node's initial data
            deadline: Optional run deadline (epoch seconds). Defaults to
                ``GraphSpec.timeout_seconds`` from run start, if set.
            grace: seconds after the deadline in which non-LLM nodes (e.g. a
                fallback reply) may still run. Default 1.0.
            
        Returns:
            Final state with results from all workers in agent_data.results.
            If the run raises and the agent has a checkpointer, the exception
            carries ``run_result`` built from the last checkpoint.
        """
        compiled_graph = self.get_compiled_graph()
        
        # Raise the recursion limit to support multi-step planning
        # loops (each orchestrator ↔ tool_executor round trip traverses
        # ~10+ graph nodes; LangGraph 1.2 counts subgraph nodes more
        # granularly so 200 is needed for complex interactive loops).
        config = {**(config or {})}
        config.setdefault("recursion_limit", DEFAULT_RECURSION_LIMIT)

        try:
            return await compiled_graph.ainvoke(
                self._initial_state(messages, start_data, deadline, grace), config=config)
        except Exception as e:
            if self.checkpointer is not None and (config.get("configurable") or {}).get("thread_id"):
                try:
                    snapshot = await compiled_graph.aget_state(config)
                    e.run_result = self._run_result(dict(snapshot.values or {}), e)
                except Exception:
                    logger.debug("Could not attach run_result from the checkpointer", exc_info=True)
            raise

    @staticmethod
    def _initial_state(messages, start_data, deadline, grace=None) -> Dict[str, Any]:
        run_input: Dict[str, Any] = {}
        if start_data is not None:
            run_input["start_data"] = start_data
        if deadline is not None:
            run_input["deadline"] = deadline
        if grace is not None:
            run_input["grace"] = grace
        return {
            "messages": messages or [],
            "agent_data": {"run_input": run_input} if run_input else {},
            "work": {},
        }

    @staticmethod
    def _run_result(state: Dict[str, Any], exc: Optional[BaseException] = None) -> RunResult:
        """A RunResult from a (possibly partial) final state. For a failed
        run, the failing node's error and its attempts' usage are added."""
        agent_data = dict(state.get("agent_data") or {})
        node_id = getattr(exc, "kg_node_id", None) if exc is not None else None
        if node_id is not None:
            update: Dict[str, Any] = {"errors": {node_id: {
                "type": error_type_of(exc),
                "message": str(exc),
                "exception": type(exc).__name__,
                "attempts": getattr(exc, "attempts", 1),
            }}}
            update.update(usage_state_update(state, node_id, getattr(exc, "attempt_usage", [])))
            agent_data = merge_agent_data(agent_data, update)
        elif isinstance(exc, TimeoutError):
            agent_data = merge_agent_data(agent_data, {"errors": {"__run__": {
                "type": "timeout",
                "message": "run deadline (plus grace) passed",
                "exception": "TimeoutError",
                "attempts": 1,
            }}})
        return RunResult(
            results=agent_data.get("results") or {},
            errors=agent_data.get("errors") or {},
            usage=agent_data.get("usage_total") or {},
            usage_log=agent_data.get("usage_log") or {},
            run=agent_data.get("run") or {},
            trace=agent_data.get("trace") or [],
            state={**state, "agent_data": agent_data},
        )

    async def run(
        self,
        start_data: Optional[Dict[str, Any]] = None,
        *,
        timeout: Optional[float] = None,
        deadline: Optional[float] = None,
        grace: float = DEFAULT_DEADLINE_GRACE,
        recursion_limit: int = DEFAULT_RECURSION_LIMIT,
        config: Optional[Dict[str, Any]] = None,
    ) -> RunResult:
        """Run the graph once, without a checkpointer, and return its outputs.

        Compiles once and reuses the graph; one agent is safe to share across
        concurrent runs. No ``thread_id`` is needed and nothing is retained
        between runs.

        Args:
            start_data: merged over the start node's initial data (over
                ``initial_data["args"]`` when that key exists); bindings read
                it as ``from_node="start"``.
            timeout: seconds. Sets the run deadline, seen by conditions and
                node budgets.
            deadline: absolute deadline (epoch seconds), instead of timeout.
            grace: seconds after the deadline in which non-LLM nodes may still
                run, e.g. a fallback that produces a canned reply. If the run
                is still going shortly after deadline + grace it is cancelled
                and the builtin ``TimeoutError`` is raised.
            recursion_limit: LangGraph recursion limit.
            config: extra LangGraph config (callbacks, tags, ...).

        Raises:
            Whatever the run raised (a worker error, ``ExecGraph*Error``,
            ``TimeoutError``, ...), with ``exc.run_result`` set to a
            ``RunResult`` of everything produced before the failure,
            including usage.
        """
        if timeout is not None:
            deadline = time.time() + timeout if deadline is None else min(deadline, time.time() + timeout)
        if deadline is None and self.graph_spec.timeout_seconds:
            deadline = time.time() + self.graph_spec.timeout_seconds
        cfg = {**(config or {}), "recursion_limit": recursion_limit}
        graph = self._get_compiled_graph_no_checkpointer()
        state_in = self._initial_state(None, start_data or {}, deadline, grace)

        backstop = contextlib.nullcontext()
        if deadline is not None:
            loop_deadline = asyncio.get_running_loop().time() + max(
                0.0, deadline + grace + BACKSTOP_MARGIN - time.time())
            backstop = asyncio.timeout_at(loop_deadline)

        last: Dict[str, Any] = state_in
        try:
            async with backstop:
                async for values in graph.astream(state_in, config=cfg, stream_mode="values"):
                    last = values
        except Exception as e:
            try:
                e.run_result = self._run_result(last, e)
            except Exception:
                logger.debug("Could not attach run_result", exc_info=True)
            raise
        return self._run_result(last)
    
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
