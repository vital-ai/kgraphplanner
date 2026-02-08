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
    merge: Literal["edge_overrides", "node_defaults_then_edge", "accumulate"] = "edge_overrides"
    condition: Optional[str] = None  # Condition expression for conditional routing


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
    timeout_seconds: Optional[int] = None
    
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
        "circular_dependency",
        "unreachable_node",
        "missing_required_input"
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

def validate_graph_spec(graph_spec: GraphSpec) -> ValidationResult:
    """Validate a graph specification for correctness."""
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
    
    # Check for invalid edges (referencing non-existent nodes)
    valid_node_ids = set(node_ids)
    for edge in graph_spec.edges:
        if edge.source not in valid_node_ids:
            errors.append(GraphValidationError(
                error_type="missing_node",
                message=f"Edge references non-existent source node: {edge.source}",
                edge_source=edge.source,
                edge_destination=edge.destination
            ))
        if edge.destination not in valid_node_ids:
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
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        info=info
    )


