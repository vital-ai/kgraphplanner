from __future__ import annotations

from typing import Dict, List, Any, Optional, Union, Literal, TypedDict
from pydantic import BaseModel, Field, ConfigDict
from typing_extensions import Annotated
from operator import or_ as merge_dicts

# ============================================================
# Core Graph Component Models
# ============================================================

class Binding(BaseModel):
    """Represents data binding between nodes in the execution graph."""
    model_config = ConfigDict(extra="forbid")
    
    from_node: Optional[str] = None
    path: str = "$"
    literal: Optional[Any] = None
    transform: Optional[Literal["as_is", "text", "json"]] = "as_is"
    reduce: Optional[Literal["overwrite", "append_list", "merge_dict", "concat_text"]] = None
    alias: Optional[str] = None

class EdgeSpec(BaseModel):
    """Specification for an edge connecting two nodes in the execution graph."""
    model_config = ConfigDict(extra="forbid")
    
    source: str
    destination: str
    prompt: Optional[str] = None
    bindings: Dict[str, List[Binding]] = Field(default_factory=dict)
    # How this edge's prompt combines with the destination's existing prompt:
    # "accumulate" appends it; the other two values replace it (they behave
    # the same). Bindings are unaffected; see WorkerNodeSpec.reentry.
    merge: Literal["edge_overrides", "node_defaults_then_edge", "accumulate"] = "edge_overrides"
    condition: Optional[str] = None  # Condition expression for conditional routing (see graph/conditions.py)
    max_traversals: Optional[int] = None  # Max times this edge can be traversed per run
    # What happens when max_traversals is reached. None resolves by source type:
    # an LLM worker gets "source_final_turn", anything else gets "error".
    #   source_final_turn: re-run the source with a "compose your final answer" prompt
    #   default:           take the source's __default__ edge
    #   error:             raise ExecGraphLoopExhausted
    on_exhausted: Optional[Literal["source_final_turn", "default", "error"]] = None


# ============================================================
# Execution State Models
# ============================================================

class Activation(TypedDict, total=False):
    """Represents the activation state for a node during execution."""
    prompt: str
    args: Dict[str, Any]

class ExecState(TypedDict, total=False):
    """State maintained during graph execution."""
    # Allow concurrent writes; values are merged like: left | right
    results: Annotated[Dict[str, Any], merge_dicts]
    errors: Annotated[Dict[str, str], merge_dicts]
    activation: Annotated[Dict[str, Activation], merge_dicts]
    work: Annotated[Dict[str, Any], merge_dicts]

# ============================================================
# Node Type Specifications
# ============================================================

class RetrySpec(BaseModel):
    """Retry policy for a worker node. Backoff never sleeps past the run deadline."""
    model_config = ConfigDict(extra="forbid")
    
    max_attempts: int = Field(2, ge=1, le=10)  # total attempts, including the first
    initial_interval: float = Field(0.5, ge=0)  # seconds before the first retry
    backoff_factor: float = Field(2.0, ge=1)
    max_interval: float = Field(30.0, ge=0)
    jitter: bool = True
    # Which failures are retried, by name (see kgraphplanner/agent/node_runtime.py):
    #   timeout, rate_limit (429), server_error (5xx), connection,
    #   client_error (4xx other than 401/403/429), auth_error (401/403),
    #   llm_error (any failed LLM call), parse_error (reply didn't validate)
    # Only the first four are retried by default.
    retry_on: List[Literal["timeout", "rate_limit", "server_error", "connection",
                           "client_error", "auth_error", "llm_error", "parse_error"]] = Field(
        default_factory=lambda: ["timeout", "rate_limit", "server_error", "connection"]
    )

class WorkerNodeSpec(BaseModel):
    """Specification for a worker node that executes tasks."""
    model_config = ConfigDict(extra="forbid")
    
    id: str
    node_type: Literal["worker"] = "worker"
    worker_name: str
    system_directive: Optional[str] = None
    tools: List[str] = Field(default_factory=list)
    required_inputs: List[str] = Field(default_factory=list)
    max_iterations: int = 6
    defaults: Dict[str, Any] = Field(default_factory=dict)
    # Failure policy, applied after retries are used up:
    #   raise:  the exception propagates out of ainvoke
    #   record: write agent_data.errors[id] and results[id] = {"status": "error", ...}
    #           and continue; route on it with an "on_error" or "eq:status:error" edge
    on_error: Literal["raise", "record"] = "raise"
    # Per-attempt budget, capped by the time left before the run deadline.
    timeout_seconds: Optional[float] = Field(None, gt=0)
    retry: Optional[RetrySpec] = None
    # How a re-entered node's inputs are built:
    #   keep:  merge new bindings into the previous activation (default)
    #   reset: start from the node defaults plus the incoming edge only
    # (Not related to EdgeSpec.merge, which only combines prompts.)
    reentry: Literal["keep", "reset"] = "keep"

class ConditionalNodeSpec(BaseModel):
    """Specification for a conditional routing node."""
    model_config = ConfigDict(extra="forbid")
    
    id: str
    node_type: Literal["conditional"] = "conditional"
    condition_function: str  # Name of the function to evaluate
    routes: Dict[str, str]  # Condition result -> destination node mapping
    defaults: Dict[str, Any] = Field(default_factory=dict)

class StartNodeSpec(BaseModel):
    """Specification for the graph entry point."""
    model_config = ConfigDict(extra="forbid")
    
    id: str = "start"
    node_type: Literal["start"] = "start"
    initial_data: Dict[str, Any] = Field(default_factory=dict)

class EndNodeSpec(BaseModel):
    """Specification for graph exit points."""
    model_config = ConfigDict(extra="forbid")
    
    id: str = "end"
    node_type: Literal["end"] = "end"
    output_transform: Optional[str] = None  # Optional transformation function

class AggregatorNodeSpec(BaseModel):
    """Specification for nodes that aggregate results from multiple sources."""
    model_config = ConfigDict(extra="forbid")
    
    id: str
    node_type: Literal["aggregator"] = "aggregator"
    aggregation_function: str  # Name of the aggregation function
    input_parameter: str = "items"
    reduce_method: Literal["append_list", "merge_dict", "concat_text"] = "append_list"
    defaults: Dict[str, Any] = Field(default_factory=dict)

# Union type for all node specifications
NodeSpecUnion = Union[
    WorkerNodeSpec,
    ConditionalNodeSpec, 
    StartNodeSpec,
    EndNodeSpec,
    AggregatorNodeSpec
]

# ============================================================
# Enhanced Graph Specification
# ============================================================

class GraphSpec(BaseModel):
    """Complete specification for an execution graph with typed nodes and execution metadata."""
    model_config = ConfigDict(extra="forbid")
    
    graph_id: str
    name: Optional[str] = None
    description: Optional[str] = None
    version: str = "1.0"
    
    # Core graph structure
    nodes: List[NodeSpecUnion]
    edges: List[EdgeSpec]
    
    # Execution configuration
    entry_points: List[str] = Field(default_factory=lambda: ["start"])
    exit_points: List[str] = Field(default_factory=lambda: ["end"])
    max_parallel_execution: int = Field(3, ge=1, le=8)
    timeout_seconds: Optional[float] = None  # default run deadline, from run start
    
    # Metadata
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    
    def get_node_by_id(self, node_id: str) -> Optional[NodeSpecUnion]:
        """Retrieve a node by its ID."""
        return next((node for node in self.nodes if node.id == node_id), None)
    
    def get_edges_from_node(self, node_id: str) -> List[EdgeSpec]:
        """Get all edges originating from a specific node."""
        return [edge for edge in self.edges if edge.source == node_id]
    
    def get_edges_to_node(self, node_id: str) -> List[EdgeSpec]:
        """Get all edges targeting a specific node."""
        return [edge for edge in self.edges if edge.destination == node_id]

# ============================================================
# Execution Context Models
# ============================================================

class ExecutionContext(BaseModel):
    """Context information for graph execution."""
    model_config = ConfigDict(extra="forbid")
    
    execution_id: str
    graph_spec: GraphSpec
    worker_registry: Dict[str, str]  # worker_name -> worker_class_path
    tool_registry: Dict[str, str]    # tool_name -> tool_class_path
    
    # Runtime configuration
    recursion_limit: int = 300
    streaming_enabled: bool = True
    checkpoint_enabled: bool = True
    thread_id: Optional[str] = None
    
    # Execution state
    current_state: Optional[Dict[str, Any]] = None
    execution_history: List[Dict[str, Any]] = Field(default_factory=list)

class ExecutionResult(BaseModel):
    """Result of graph execution."""
    model_config = ConfigDict(extra="forbid")
    
    execution_id: str
    success: bool
    final_state: Dict[str, Any]
    errors: Dict[str, str] = Field(default_factory=dict)
    execution_time_seconds: Optional[float] = None
    nodes_executed: List[str] = Field(default_factory=list)
    
    # Output data
    results: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

# ============================================================
# Graph Validation Models
# ============================================================

class GraphValidationError(BaseModel):
    """Represents a validation error in the graph specification."""
    model_config = ConfigDict(extra="forbid")
    
    error_type: Literal[
        "missing_node", 
        "duplicate_node", 
        "invalid_edge", 
        "missing_worker", 
        "unreachable_node",
        "missing_required_input",
        "invalid_condition",
        "missing_default",
        "mixed_routing",
        "invalid_option",
        "unhandled_error",
    ]
    message: str
    node_id: Optional[str] = None
    edge_source: Optional[str] = None
    edge_destination: Optional[str] = None
    severity: Literal["error", "warning", "info"] = "error"

class ValidationResult(BaseModel):
    """Result of graph validation."""
    model_config = ConfigDict(extra="forbid")
    
    is_valid: bool
    errors: List[GraphValidationError] = Field(default_factory=list)
    warnings: List[GraphValidationError] = Field(default_factory=list)
    info: List[GraphValidationError] = Field(default_factory=list)
    
    def has_errors(self) -> bool:
        """Check if there are any validation errors."""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if there are any validation warnings."""
        return len(self.warnings) > 0

# ============================================================
# Utility Functions
# ============================================================

def create_simple_graph_spec(
    nodes: List[Dict[str, Any]], 
    edges: List[Dict[str, str]],
    graph_id: str = "default"
) -> GraphSpec:
    """Create a simple graph specification from basic node and edge definitions."""
    
    # Convert node dictionaries to proper node specs
    node_specs = []
    for node_dict in nodes:
        node_type = node_dict.get("node_type", "worker")
        
        if node_type == "worker":
            node_specs.append(WorkerNodeSpec(**node_dict))
        elif node_type == "conditional":
            node_specs.append(ConditionalNodeSpec(**node_dict))
        elif node_type == "start":
            node_specs.append(StartNodeSpec(**node_dict))
        elif node_type == "end":
            node_specs.append(EndNodeSpec(**node_dict))
        elif node_type == "aggregator":
            node_specs.append(AggregatorNodeSpec(**node_dict))
        else:
            # Default to worker node
            node_specs.append(WorkerNodeSpec(
                id=node_dict["id"],
                worker_name=node_dict.get("worker", node_dict["id"]),
                **{k: v for k, v in node_dict.items() if k not in ["id", "worker"]}
            ))
    
    # Convert edge dictionaries to EdgeSpec objects
    edge_specs = [EdgeSpec(**edge_dict) for edge_dict in edges]
    
    return GraphSpec(
        graph_id=graph_id,
        nodes=node_specs,
        edges=edge_specs
    )

def _validate_start_guards(edges, errors, warnings, allow_expressions: bool) -> None:
    """Start-edge conditions are guards: each fires independently, so the
    mixed-routing and missing-default rules for worker sources don't apply."""
    from kgraphplanner.graph.conditions import parse_condition
    from kgraphplanner.graph.exec_errors import ConditionSyntaxError

    defaults = []
    for e in edges:
        if e.max_traversals is not None or e.on_exhausted is not None:
            errors.append(GraphValidationError(
                error_type="invalid_option",
                message=(f"Start edge to '{e.destination}' sets max_traversals/on_exhausted; "
                         f"start edges don't loop"),
                edge_source=e.source, edge_destination=e.destination))
        if e.condition is None:
            continue
        try:
            cond = parse_condition(e.condition, source="start", allow_expressions=allow_expressions)
        except ConditionSyntaxError as ce:
            errors.append(GraphValidationError(
                error_type="invalid_condition", message=str(ce),
                edge_source=e.source, edge_destination=e.destination))
            continue
        if cond.kind == "on_error":
            errors.append(GraphValidationError(
                error_type="invalid_condition",
                message=f"Start edge to '{e.destination}' uses on_error; start has no node to fail",
                edge_source=e.source, edge_destination=e.destination))
        if cond.is_default:
            defaults.append(e)
    if len(defaults) > 1:
        errors.append(GraphValidationError(
            error_type="invalid_edge",
            message=f"start has {len(defaults)} __default__ edges; at most one is allowed",
            node_id="start"))
    if defaults and any(e.condition is None for e in edges):
        warnings.append(GraphValidationError(
            error_type="invalid_option",
            message=("start has a __default__ edge and unconditional edges; the default "
                     "fires only when no other start edge does, so it can never fire"),
            node_id="start", severity="warning"))


def validate_graph_spec(graph_spec: GraphSpec, *, allow_expressions: bool = True) -> ValidationResult:
    """Validate a graph specification for correctness.

    Covers structure (duplicate or missing nodes, unreachable nodes) and
    routing: conditions that don't parse, sources with no ``__default__``,
    sources that mix conditional and unconditional out-edges, and node options
    that can't work. Checks that need the worker registry live in
    ``KGraphExecGraphAgent``.
    """
    from collections import defaultdict
    from kgraphplanner.graph.conditions import parse_condition
    from kgraphplanner.graph.exec_errors import ConditionSyntaxError

    errors = []
    warnings = []
    info = []
    
    # Check for duplicate node IDs
    node_ids = [node.id for node in graph_spec.nodes]
    duplicates = set([nid for nid in node_ids if node_ids.count(nid) > 1])
    for dup_id in duplicates:
        errors.append(GraphValidationError(
            error_type="duplicate_node",
            message=f"Duplicate node ID: {dup_id}",
            node_id=dup_id
        ))
    
    # Check for invalid edges (referencing non-existent nodes).
    # "start" and "end" are always valid: the exec graph agent handles them
    # whether or not the spec declares Start/End nodes.
    valid_node_ids = set(node_ids)
    known_ids = valid_node_ids | {"start", "end"}
    for edge in graph_spec.edges:
        if edge.source not in known_ids:
            errors.append(GraphValidationError(
                error_type="missing_node",
                message=f"Edge references non-existent source node: {edge.source}",
                edge_source=edge.source,
                edge_destination=edge.destination
            ))
        if edge.destination not in known_ids:
            errors.append(GraphValidationError(
                error_type="missing_node",
                message=f"Edge references non-existent destination node: {edge.destination}",
                edge_source=edge.source,
                edge_destination=edge.destination
            ))
    
    # Check for unreachable nodes
    reachable = set(graph_spec.entry_points)
    changed = True
    while changed:
        changed = False
        for edge in graph_spec.edges:
            if edge.source in reachable and edge.destination not in reachable:
                reachable.add(edge.destination)
                changed = True
    
    unreachable = valid_node_ids - reachable
    for node_id in unreachable:
        warnings.append(GraphValidationError(
            error_type="unreachable_node",
            message=f"Node is unreachable from entry points: {node_id}",
            node_id=node_id,
            severity="warning"
        ))
    
    # --- Routing checks ---
    out_edges = defaultdict(list)
    for edge in graph_spec.edges:
        out_edges[edge.source].append(edge)

    for source, edges in out_edges.items():
        if source == "start":
            _validate_start_guards(edges, errors, warnings, allow_expressions)
            continue
        conditional = [e for e in edges if e.condition is not None]
        if not conditional:
            for e in edges:
                if e.max_traversals is not None or e.on_exhausted is not None:
                    warnings.append(GraphValidationError(
                        error_type="invalid_option",
                        message=(f"Edge '{e.source}' -> '{e.destination}' sets max_traversals/"
                                 f"on_exhausted but has no condition; they only apply to "
                                 f"conditional edges"),
                        edge_source=e.source, edge_destination=e.destination, severity="warning"))
            continue

        for e in conditional:
            try:
                parse_condition(e.condition, source=source, allow_expressions=allow_expressions)
            except ConditionSyntaxError as ce:
                errors.append(GraphValidationError(
                    error_type="invalid_condition", message=str(ce),
                    edge_source=e.source, edge_destination=e.destination))

        unconditional = [e for e in edges if e.condition is None]
        if unconditional:
            errors.append(GraphValidationError(
                error_type="mixed_routing",
                message=(f"'{source}' has both conditional and unconditional out-edges "
                         f"(unconditional to {[e.destination for e in unconditional]}); "
                         f"both would fire. Make them all conditional, using __default__ "
                         f"for the fallback"),
                node_id=source))

        defaults = [e for e in conditional if e.condition.strip() == "__default__"]
        if len(defaults) > 1:
            errors.append(GraphValidationError(
                error_type="invalid_edge",
                message=f"'{source}' has {len(defaults)} __default__ edges; at most one is allowed",
                node_id=source))
        exhaustive = any(e.condition.strip() in ("always", "true") for e in conditional)
        if not defaults and not exhaustive:
            warnings.append(GraphValidationError(
                error_type="missing_default",
                message=(f"'{source}' has conditional edges but no __default__ edge; "
                         f"a run where no condition matches raises ExecGraphRoutingError"),
                node_id=source, severity="warning"))

        for e in conditional:
            if e.on_exhausted is not None and e.max_traversals is None:
                warnings.append(GraphValidationError(
                    error_type="invalid_option",
                    message=(f"Edge '{e.source}' -> '{e.destination}' sets on_exhausted "
                             f"without max_traversals; it has no effect"),
                    edge_source=e.source, edge_destination=e.destination, severity="warning"))
            if e.on_exhausted == "default" and not defaults:
                errors.append(GraphValidationError(
                    error_type="missing_default",
                    message=(f"Edge '{e.source}' -> '{e.destination}' has "
                             f"on_exhausted='default' but '{source}' has no __default__ edge"),
                    edge_source=e.source, edge_destination=e.destination))

    for node in graph_spec.nodes:
        if isinstance(node, WorkerNodeSpec) and node.on_error == "record":
            handled = any(
                e.condition is not None
                and e.condition.strip() in ("on_error", "eq:status:error", "has:error_type")
                for e in out_edges.get(node.id, [])
            )
            if not handled:
                warnings.append(GraphValidationError(
                    error_type="unhandled_error",
                    message=(f"Node '{node.id}' has on_error='record' but no out-edge "
                             f"routes on the error (on_error or eq:status:error); a "
                             f"recorded error will flow on as if the step succeeded"),
                    node_id=node.id, severity="warning"))

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        info=info
    )


