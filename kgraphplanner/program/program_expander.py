from __future__ import annotations

import logging
from typing import Dict, List, Any, Optional, Tuple

from kgraphplanner.program.program import (
    ProgramSpec, StaticNodeSpec, StaticBinding, TemplateSpec,
    ForEachSpec, NodePattern, EdgePattern, FanOutSpec, FanInSpec,
    FanOutBranch, BindingPattern, NodeDefaults, ConditionalSpec
)
from kgraphplanner.graph.exec_graph import (
    GraphSpec, EdgeSpec, Binding,
    WorkerNodeSpec, StartNodeSpec, EndNodeSpec,
    validate_graph_spec, ValidationResult
)

logger = logging.getLogger(__name__)


def validate_program_spec(program: ProgramSpec) -> Tuple[bool, List[str]]:
    """
    Validate a ProgramSpec for correctness.
    
    Returns:
        Tuple of (is_valid, list of error/warning messages)
    """
    messages = []
    
    if not program.program_id:
        messages.append("ERROR: Program ID is required")
        return False, messages
    
    if not program.static_nodes:
        messages.append("ERROR: At least one static node is required")
        return False, messages
    
    node_ids = [node.id for node in program.static_nodes]
    if "start" not in node_ids:
        messages.append("ERROR: 'start' node is required")
        return False, messages
    
    if "end" not in node_ids:
        messages.append("ERROR: 'end' node is required")
        return False, messages
    
    for template in program.templates:
        if not template.name:
            messages.append("ERROR: Template missing name")
            return False, messages
        
        if not template.for_each:
            messages.append(f"ERROR: Template '{template.name}' missing for_each specification")
            return False, messages
        
        if not (template.loop_nodes or template.fan_out or template.fan_in):
            messages.append(f"ERROR: Template '{template.name}' has no expandable content")
            return False, messages
    
    if program.required_workers:
        template_workers = set()
        for template in program.templates:
            for node_pattern in template.loop_nodes:
                template_workers.add(node_pattern.worker)
        
        static_workers = {node.worker for node in program.static_nodes}
        all_workers = template_workers | static_workers
        
        missing_workers = set(program.required_workers) - all_workers
        if missing_workers:
            messages.append(f"WARNING: Required workers not found in program: {missing_workers}")
    
    # Check for worker→worker static edges missing bindings
    non_terminal = {n.id for n in program.static_nodes if n.worker not in ("start", "end")}
    for edge in program.static_edges:
        if edge.source in non_terminal and edge.destination in non_terminal and not edge.bindings:
            messages.append(
                f"ERROR: Static edge '{edge.source}' -> '{edge.destination}' "
                f"has no bindings. Downstream worker won't receive upstream results. "
                f"Add bindings like: {{\"input\": [{{\"from_node\": \"{edge.source}\", \"path\": \"$.result_text\"}}]}}"
            )
            return False, messages
    
    # Check for redundant start→downstream loop_edges that bypass required flow.
    # If a node is already a fan_out destination, a loop_edge from "start" to
    # it would fire before the upstream worker completes, skipping data flow.
    for template in program.templates:
        fan_out_dests = set()
        for fo in template.fan_out:
            for branch in fo.branches:
                fan_out_dests.add(branch.destination_tpl)
        for le in template.loop_edges:
            if le.source_tpl == "start" and le.destination_tpl in fan_out_dests:
                messages.append(
                    f"ERROR: Template '{template.name}' has a redundant loop_edge "
                    f"'start' -> '{le.destination_tpl}' which is already reached via "
                    f"fan_out. This edge would bypass the required upstream worker. "
                    f"Remove this loop_edge — use loop_edges only for start -> "
                    f"the first step in the loop (e.g., start -> research_{{idx}})."
                )
                return False, messages
    
    # Check for missing bindings on template loop_edges and fan_out branches
    for template in program.templates:
        # loop_edges: any edge whose source is not "start" and dest is not "end" needs bindings
        for le in template.loop_edges:
            src = le.source_tpl
            dst = le.destination_tpl
            if src != "start" and dst != "end" and not le.bindings:
                messages.append(
                    f"ERROR: Template '{template.name}' loop_edge '{src}' -> '{dst}' "
                    f"has no bindings. Downstream worker won't receive upstream results. "
                    f"Add bindings like: {{\"input\": [{{\"from_node_tpl\": \"{src}\", \"path\": \"$.result_text\"}}]}}"
                )
                return False, messages
        
        # fan_out branches: every branch needs bindings to forward the source's output
        for fo in template.fan_out:
            for branch in fo.branches:
                if not branch.bindings:
                    messages.append(
                        f"ERROR: Template '{template.name}' fan_out branch "
                        f"'{fo.source_tpl}' -> '{branch.destination_tpl}' has no bindings. "
                        f"Each fan_out branch MUST include bindings to pass the source's result. "
                        f"Add: {{\"input\": [{{\"from_node_tpl\": \"{fo.source_tpl}\", \"path\": \"$.result_text\"}}]}}"
                    )
                    return False, messages
    
    return True, messages


def _resolve_path_simple(root: Any, path: str) -> Any:
    """Simple JSONPath-like resolution for start_args paths."""
    if path in (None, "", "$"):
        return root
    p = path
    if p.startswith("$."):
        p = p[2:]
    elif p.startswith("$"):
        p = p[1:]
    if not p:
        return root
    return root.get(p) if isinstance(root, dict) else None


def _format_binding_patterns(
    binding_patterns: Dict[str, List[BindingPattern]],
    context: Dict[str, Any]
) -> Dict[str, List[Binding]]:
    """Convert BindingPatterns to Bindings by applying template context."""
    bindings = {}
    for param, patterns in binding_patterns.items():
        binding_list = []
        for bp in patterns:
            from_node = bp.from_node_tpl.format(**context) if bp.from_node_tpl else None
            path = bp.path.format(**context) if bp.path else "$"
            binding_list.append(Binding(
                from_node=from_node,
                path=path,
                transform=bp.transform,
                reduce=bp.reduce,
                alias=bp.alias
            ))
        bindings[param] = binding_list
    return bindings


def _format_defaults(defaults_tpl: Optional[NodeDefaults], context: Dict[str, Any]) -> Dict[str, Any]:
    """Format node defaults template with context variables."""
    if not defaults_tpl:
        return {}
    
    defaults = {}
    defaults_dict = defaults_tpl.model_dump()
    
    if defaults_dict.get("args"):
        formatted_args = {}
        for k, v in defaults_dict["args"].items():
            if isinstance(v, str):
                formatted_args[k] = v.format(**context)
            else:
                formatted_args[k] = v
        defaults["args"] = formatted_args
    
    if defaults_dict.get("prompt"):
        defaults["prompt"] = defaults_dict["prompt"].format(**context)
    
    return defaults


def expand_program_to_graph(
    program: ProgramSpec,
    start_seed: Optional[Dict[str, Any]] = None
) -> GraphSpec:
    """
    Convert a ProgramSpec to a GraphSpec by expanding templates.
    
    Args:
        program: The ProgramSpec to expand
        start_seed: Optional additional args to merge into start node args
        
    Returns:
        Expanded GraphSpec ready for execution
    """
    # Convert static nodes to typed GraphSpec nodes
    nodes = []
    for static_node in program.static_nodes:
        if static_node.id == "start":
            nodes.append(StartNodeSpec(
                id=static_node.id,
                initial_data=static_node.defaults.model_dump() if static_node.defaults else {}
            ))
        elif static_node.id == "end":
            nodes.append(EndNodeSpec(id=static_node.id))
        else:
            nodes.append(WorkerNodeSpec(
                id=static_node.id,
                worker_name=static_node.worker,
                defaults=static_node.defaults.model_dump() if static_node.defaults else {}
            ))
    
    # Convert static edges
    edges = []
    for static_edge in program.static_edges:
        # Convert StaticBinding → Binding
        edge_bindings = {}
        for param, sb_list in static_edge.bindings.items():
            edge_bindings[param] = [
                Binding(
                    from_node=sb.from_node,
                    path=sb.path,
                    literal=sb.literal,
                    transform=sb.transform,
                    reduce=sb.reduce,
                    alias=sb.alias
                ) for sb in sb_list
            ]
        edges.append(EdgeSpec(
            source=static_edge.source,
            destination=static_edge.destination,
            prompt=static_edge.prompt,
            bindings=edge_bindings
        ))
    
    # Get start args for template expansion
    start_node = program.get_static_node_by_id("start")
    start_args = {}
    if start_node and start_node.defaults and start_node.defaults.args:
        start_args = dict(start_node.defaults.args)
    if start_seed:
        start_args = {**start_args, **start_seed}
    
    # Expand templates
    for template in program.templates:
        logger.info(f"Expanding template: {template.name}")
        
        # Resolve items to iterate over
        if template.for_each.source_from == "literal":
            items = template.for_each.literal_items or []
        else:
            items = _resolve_path_simple(start_args, template.for_each.source_path) or []
        
        logger.info(f"  Processing {len(items)} items")
        
        for idx, item in enumerate(items):
            context = {
                template.for_each.item_var: item,
                template.for_each.idx_var: idx,
                **{k: v for k, v in start_args.items() if isinstance(v, (str, int, float, bool))}
            }
            # If item is a dict, flatten its string keys into context and
            # wrap in SimpleNamespace so {item_var.key} dot-access works.
            if isinstance(item, dict):
                from types import SimpleNamespace
                context[template.for_each.item_var] = SimpleNamespace(**item)
                for k, v in item.items():
                    if isinstance(v, str):
                        context.setdefault(k, v)
            
            # Expand loop nodes
            for node_pattern in template.loop_nodes:
                node_id = node_pattern.id_tpl.format(**context)
                defaults = _format_defaults(node_pattern.defaults_tpl, context)
                
                nodes.append(WorkerNodeSpec(
                    id=node_id,
                    worker_name=node_pattern.worker,
                    defaults=defaults
                ))
            
            # Expand loop edges
            for edge_pattern in template.loop_edges:
                source = edge_pattern.source_tpl.format(**context)
                destination = edge_pattern.destination_tpl.format(**context)
                prompt = edge_pattern.prompt_tpl.format(**context) if edge_pattern.prompt_tpl else None
                bindings = _format_binding_patterns(edge_pattern.bindings, context)
                
                edges.append(EdgeSpec(
                    source=source,
                    destination=destination,
                    prompt=prompt,
                    bindings=bindings
                ))
            
            # Expand fan-out
            for fan_out in template.fan_out:
                source = fan_out.source_tpl.format(**context)
                for branch in fan_out.branches:
                    destination = branch.destination_tpl.format(**context)
                    prompt = branch.prompt_tpl.format(**context) if branch.prompt_tpl else None
                    bindings = _format_binding_patterns(branch.bindings, context)
                    
                    edges.append(EdgeSpec(
                        source=source,
                        destination=destination,
                        prompt=prompt,
                        bindings=bindings
                    ))
            
            # Expand fan-in
            for fan_in in template.fan_in:
                destination = fan_in.destination_tpl.format(**context)
                prompt = fan_in.prompt_tpl.format(**context) if fan_in.prompt_tpl else None
                
                for source_tpl in fan_in.sources_tpl:
                    source = source_tpl.format(**context)
                    
                    bindings = {fan_in.param: [Binding(
                        from_node=source,
                        path="$",
                        reduce=fan_in.reduce
                    )]}
                    
                    edges.append(EdgeSpec(
                        source=source,
                        destination=destination,
                        prompt=prompt,
                        bindings=bindings,
                        merge="accumulate"
                    ))
    
    # Expand conditionals (program-level conditional routing)
    for cond_spec in program.conditionals:
        source = cond_spec.source_tpl
        logger.info(f"Expanding conditional from '{source}' "
                    f"({len(cond_spec.branches)} branches)")
        
        for branch in cond_spec.branches:
            destination = branch.destination_tpl
            bindings = _format_binding_patterns(branch.bindings, {})
            
            edges.append(EdgeSpec(
                source=source,
                destination=destination,
                prompt=branch.prompt_tpl,
                bindings=bindings,
                condition=branch.condition
            ))
        
        if cond_spec.default_destination_tpl:
            edges.append(EdgeSpec(
                source=source,
                destination=cond_spec.default_destination_tpl,
                condition="__default__"
            ))
    
    graph_spec = GraphSpec(
        graph_id=program.program_id,
        name=program.name,
        description=program.description,
        version=program.version,
        nodes=nodes,
        edges=edges,
        exit_points=program.exit_nodes,
        max_parallel_execution=program.max_parallel
    )
    
    logger.info(f"Generated graph with {len(nodes)} nodes and {len(edges)} edges")
    return graph_spec
