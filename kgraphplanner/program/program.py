
from __future__ import annotations

from typing import Dict, List, Any, Optional, Union, Literal
from pydantic import BaseModel, Field, ConfigDict

# ============================================================
# JSON-Safe Type Definitions
# ============================================================

# Allowed scalar JSON values in program specifications
JsonScalar = Union[str, int, float, bool, None]
# Extended arg value set for program arguments — permissive to accept
# nested structures the LLM may produce (lists of dicts, etc.)
JsonArgValue = Union[JsonScalar, List[Any], Dict[str, Any]]
JsonArgs = Dict[str, JsonArgValue]

# ============================================================
# Core Program Component Models
# ============================================================

class NodeDefaults(BaseModel):
    """Default configuration for a node in the program."""
    model_config = ConfigDict(extra="forbid")
    
    prompt: Optional[str] = None
    args: Optional[JsonArgs] = None

class StaticNodeSpec(BaseModel):
    """Specification for a static node in the program."""
    model_config = ConfigDict(extra="forbid")
    
    id: str
    worker: str
    defaults: Optional[NodeDefaults] = None

class StaticBinding(BaseModel):
    """Concrete data binding for static edges (no template expansion)."""
    model_config = ConfigDict(extra="forbid")
    
    from_node: Optional[str] = None
    path: str = "$"
    literal: Optional[JsonScalar] = None
    transform: Optional[Literal["as_is", "text", "json"]] = "as_is"
    reduce: Optional[Literal["overwrite", "append_list", "merge_dict", "concat_text"]] = None
    alias: Optional[str] = None

class StaticEdgeSpec(BaseModel):
    """Specification for a static edge in the program."""
    model_config = ConfigDict(extra="forbid")
    
    source: str
    destination: str
    prompt: Optional[str] = None
    bindings: Dict[str, List[StaticBinding]] = Field(default_factory=dict)

# ============================================================
# Template Pattern Models
# ============================================================

class BindingPattern(BaseModel):
    """Pattern for data binding between nodes with template support."""
    model_config = ConfigDict(extra="forbid")
    
    from_node_tpl: Optional[str] = None
    path: str = "$"
    literal_tpl: Optional[JsonScalar] = None
    transform: Optional[Literal["as_is", "text", "json"]] = "as_is"
    reduce: Optional[Literal["overwrite", "append_list", "merge_dict", "concat_text"]] = None
    alias: Optional[str] = None

class EdgePattern(BaseModel):
    """Pattern for edges that will be expanded in templates."""
    model_config = ConfigDict(extra="forbid")
    
    source_tpl: str
    destination_tpl: str
    prompt_tpl: Optional[str] = None
    merge: Literal["edge_overrides", "node_defaults_then_edge", "accumulate"] = "edge_overrides"
    bindings: Dict[str, List[BindingPattern]] = Field(default_factory=dict)

class NodePattern(BaseModel):
    """Pattern for nodes that will be expanded in templates."""
    model_config = ConfigDict(extra="forbid")
    
    id_tpl: str
    worker: str
    defaults_tpl: Optional[NodeDefaults] = None

# ============================================================
# Loop and Iteration Models
# ============================================================

class ForEachSpec(BaseModel):
    """Specification for iterating over a collection to generate multiple nodes/edges."""
    model_config = ConfigDict(extra="forbid")
    
    source_from: Literal["start_args", "literal"] = "start_args"
    source_path: str = "$.items"
    literal_items: Optional[List[JsonScalar]] = None
    item_var: str = "item"
    idx_var: str = "idx"

# ============================================================
# Fan-Out and Fan-In Models
# ============================================================

class FanOutBranch(BaseModel):
    """A single branch in a fan-out operation."""
    model_config = ConfigDict(extra="forbid")
    
    destination_tpl: str
    prompt_tpl: Optional[str] = None
    bindings: Dict[str, List[BindingPattern]] = Field(default_factory=dict)
    merge: Literal["edge_overrides", "node_defaults_then_edge", "accumulate"] = "edge_overrides"

class FanOutSpec(BaseModel):
    """Specification for fan-out operation (one source to multiple destinations)."""
    model_config = ConfigDict(extra="forbid")
    
    source_tpl: str
    branches: List[FanOutBranch]
    description: Optional[str] = None

class FanInSpec(BaseModel):
    """Specification for fan-in operation (multiple sources to one destination)."""
    model_config = ConfigDict(extra="forbid")
    
    sources_tpl: List[str]
    destination_tpl: str
    param: str = "items"
    prompt_tpl: Optional[str] = None
    reduce: Literal["append_list", "merge_dict", "concat_text"] = "append_list"
    description: Optional[str] = None

# ============================================================
# Parallel Processing Models
# ============================================================

class ParallelProcessSpec(BaseModel):
    """Specification for parallel processing of items."""
    model_config = ConfigDict(extra="forbid")
    
    name: str = "parallel_process"
    items_source: ForEachSpec
    worker_pattern: NodePattern
    input_binding: Optional[BindingPattern] = None
    output_aggregation: Optional[FanInSpec] = None
    max_parallel: int = Field(3, ge=1, le=10)

class PipelineStageSpec(BaseModel):
    """Specification for a pipeline stage with sequential processing."""
    model_config = ConfigDict(extra="forbid")
    
    name: str
    worker: str
    input_from: Optional[str] = None
    output_to: Optional[str] = None
    transform: Optional[BindingPattern] = None

class PipelineSpec(BaseModel):
    """Specification for a sequential pipeline of processing stages."""
    model_config = ConfigDict(extra="forbid")
    
    name: str = "pipeline"
    stages: List[PipelineStageSpec]
    input_source: str = "start"
    output_destination: str = "end"

# ============================================================
# Template Specification
# ============================================================

class TemplateSpec(BaseModel):
    """Template specification for generating dynamic graph structures."""
    model_config = ConfigDict(extra="forbid")
    
    name: str = "template"
    description: Optional[str] = None
    
    # Core iteration specification
    for_each: ForEachSpec
    
    # Node and edge patterns
    loop_nodes: List[NodePattern] = Field(default_factory=list)
    loop_edges: List[EdgePattern] = Field(default_factory=list)
    
    # Fan-out and fan-in operations
    fan_out: List[FanOutSpec] = Field(default_factory=list)
    fan_in: List[FanInSpec] = Field(default_factory=list)
    
    # High-level patterns
    parallel_processes: List[ParallelProcessSpec] = Field(default_factory=list)
    pipelines: List[PipelineSpec] = Field(default_factory=list)

# ============================================================
# Conditional Processing Models
# ============================================================

class ConditionalBranch(BaseModel):
    """A conditional branch with condition and destination."""
    model_config = ConfigDict(extra="forbid")
    
    condition: str  # Expression or function name
    destination_tpl: str
    prompt_tpl: Optional[str] = None
    bindings: Dict[str, List[BindingPattern]] = Field(default_factory=dict)

class ConditionalSpec(BaseModel):
    """Specification for conditional routing in the program."""
    model_config = ConfigDict(extra="forbid")
    
    source_tpl: str
    branches: List[ConditionalBranch]
    default_destination_tpl: Optional[str] = None
    evaluation_context: Dict[str, Any] = Field(default_factory=dict)

# ============================================================
# Main Program Specification
# ============================================================

class ProgramSpec(BaseModel):
    """Complete program specification that can be expanded into a graph."""
    model_config = ConfigDict(extra="forbid")
    
    # Metadata
    program_id: str
    name: Optional[str] = None
    description: Optional[str] = None
    version: str = "1.0"
    
    # Static components
    static_nodes: List[StaticNodeSpec] = Field(default_factory=list)
    static_edges: List[StaticEdgeSpec] = Field(default_factory=list)
    
    # Dynamic components
    templates: List[TemplateSpec] = Field(default_factory=list)
    conditionals: List[ConditionalSpec] = Field(default_factory=list)
    
    # Execution configuration
    exit_nodes: List[str] = Field(default_factory=list)
    max_parallel: int = Field(3, ge=1, le=8)
    
    # Program-level configuration
    global_context: Dict[str, Any] = Field(default_factory=dict)
    required_workers: List[str] = Field(default_factory=list)
    
    def get_static_node_by_id(self, node_id: str) -> Optional[StaticNodeSpec]:
        """Get a static node by its ID."""
        return next((node for node in self.static_nodes if node.id == node_id), None)
    
    def get_template_by_name(self, template_name: str) -> Optional[TemplateSpec]:
        """Get a template by its name."""
        return next((tmpl for tmpl in self.templates if tmpl.name == template_name), None)

# ============================================================
# Utility Functions for Common Patterns
# ============================================================

def create_document_processing_program(
    documents: List[str],
    processing_worker: str,
    aggregation_worker: str = "aggregator",
    program_id: str = "doc_processing"
) -> ProgramSpec:
    """Create a program for processing multiple documents in parallel."""
    
    return ProgramSpec(
        program_id=program_id,
        name="Document Processing Program",
        description=f"Process {len(documents)} documents in parallel and aggregate results",
        
        static_nodes=[
            StaticNodeSpec(
                id="start",
                worker="start",
                defaults=NodeDefaults(args={"documents": documents})
            ),
            StaticNodeSpec(id="aggregator", worker=aggregation_worker),
            StaticNodeSpec(id="end", worker="end")
        ],
        
        static_edges=[
            StaticEdgeSpec(source="aggregator", destination="end")
        ],
        
        templates=[
            TemplateSpec(
                name="document_processing",
                description="Process each document individually",
                for_each=ForEachSpec(
                    source_from="start_args",
                    source_path="$.documents",
                    item_var="document",
                    idx_var="doc_idx"
                ),
                loop_nodes=[
                    NodePattern(
                        id_tpl="process_doc_{doc_idx}",
                        worker=processing_worker,
                        defaults_tpl=NodeDefaults(args={"document": "{document}"})
                    )
                ],
                loop_edges=[
                    EdgePattern(
                        source_tpl="start",
                        destination_tpl="process_doc_{doc_idx}",
                        prompt_tpl="Process document: {document}",
                        bindings={
                            "document": [BindingPattern(
                                from_node_tpl="start",
                                path="$.documents[{doc_idx}]"
                            )]
                        }
                    )
                ],
                fan_in=[
                    FanInSpec(
                        sources_tpl=["process_doc_{doc_idx}"],
                        destination_tpl="aggregator",
                        param="processed_documents",
                        reduce="append_list",
                        prompt_tpl="Aggregate processed document"
                    )
                ]
            )
        ],
        
        exit_nodes=["end"],
        max_parallel=3
    )

def create_pipeline_program(
    stages: List[str],
    input_data: Dict[str, Any],
    program_id: str = "pipeline"
) -> ProgramSpec:
    """Create a sequential pipeline program."""
    
    static_nodes = [
        StaticNodeSpec(
            id="start",
            worker="start",
            defaults=NodeDefaults(args=input_data)
        )
    ]
    
    static_edges = []
    
    # Create nodes for each stage
    for i, stage_worker in enumerate(stages):
        stage_id = f"stage_{i}_{stage_worker}"
        static_nodes.append(StaticNodeSpec(id=stage_id, worker=stage_worker))
        
        # Connect to previous stage
        prev_id = "start" if i == 0 else f"stage_{i-1}_{stages[i-1]}"
        static_edges.append(StaticEdgeSpec(
            source=prev_id,
            destination=stage_id,
            prompt=f"Execute stage {i+1}: {stage_worker}"
        ))
    
    # Add end node
    static_nodes.append(StaticNodeSpec(id="end", worker="end"))
    if stages:
        final_stage = f"stage_{len(stages)-1}_{stages[-1]}"
        static_edges.append(StaticEdgeSpec(source=final_stage, destination="end"))
    
    return ProgramSpec(
        program_id=program_id,
        name="Sequential Pipeline Program",
        description=f"Sequential pipeline with {len(stages)} stages",
        static_nodes=static_nodes,
        static_edges=static_edges,
        exit_nodes=["end"]
    )

def create_fan_out_fan_in_program(
    source_data: List[Any],
    processing_workers: List[str],
    aggregation_worker: str = "aggregator",
    program_id: str = "fan_out_in"
) -> ProgramSpec:
    """Create a program with fan-out to multiple workers and fan-in aggregation."""
    
    return ProgramSpec(
        program_id=program_id,
        name="Fan-Out Fan-In Program",
        description="Fan out to multiple processors, then aggregate results",
        
        static_nodes=[
            StaticNodeSpec(
                id="start",
                worker="start",
                defaults=NodeDefaults(args={"items": source_data})
            ),
            StaticNodeSpec(id="aggregator", worker=aggregation_worker),
            StaticNodeSpec(id="end", worker="end")
        ],
        
        static_edges=[
            StaticEdgeSpec(source="aggregator", destination="end")
        ],
        
        templates=[
            TemplateSpec(
                name="fan_out_processing",
                for_each=ForEachSpec(
                    source_from="start_args",
                    source_path="$.items",
                    item_var="item",
                    idx_var="idx"
                ),
                fan_out=[
                    FanOutSpec(
                        source_tpl="start",
                        branches=[
                            FanOutBranch(
                                destination_tpl=f"worker_{worker}_{idx}",
                                prompt_tpl=f"Process with {worker}: {{item}}",
                                bindings={
                                    "input": [BindingPattern(
                                        from_node_tpl="start",
                                        path="$.items[{idx}]"
                                    )]
                                }
                            ) for worker in processing_workers
                        ]
                    )
                ],
                loop_nodes=[
                    NodePattern(
                        id_tpl=f"worker_{worker}_{{idx}}",
                        worker=worker
                    ) for worker in processing_workers
                ],
                fan_in=[
                    FanInSpec(
                        sources_tpl=[f"worker_{worker}_{{idx}}" for worker in processing_workers],
                        destination_tpl="aggregator",
                        param="results",
                        reduce="append_list"
                    )
                ]
            )
        ],
        
        exit_nodes=["end"]
    )


