from __future__ import annotations

import logging
from typing import Dict, List, Any, Optional, TypedDict

from kgraphplanner.graph.exec_graph import EdgeSpec, Binding, Activation

logger = logging.getLogger(__name__)


def get_by_path(obj: Any, path: str) -> Any:
    """
    Resolve a JSONPath-like expression against a data object.
    
    Supports:
        - "$" or "" or None -> return root object
        - "$.key" -> dict lookup
        - "$.key.nested" -> nested dict lookup
        - "$.key[0]" -> list index access
    
    Args:
        obj: The root data object to resolve against
        path: JSONPath-like expression
        
    Returns:
        Resolved value, or None if path cannot be resolved
    """
    if path in (None, "", "$"):
        return obj
    cur = obj
    p = path
    if p.startswith("$"):
        p = p[1:]
    if p.startswith("."):
        p = p[1:]
    if not p:
        return cur
    for part in p.split("."):
        if "[" in part and part.endswith("]"):
            key, idx_str = part[:-1].split("[")
            if key:
                cur = cur.get(key, None) if isinstance(cur, dict) else None
            if cur is None or not isinstance(cur, list):
                return None
            try:
                cur = cur[int(idx_str)]
            except (ValueError, IndexError):
                return None
        else:
            cur = cur.get(part, None) if isinstance(cur, dict) else None
        if cur is None:
            return None
    return cur


def apply_reduce(
    dst_args: Dict[str, Any],
    param: str,
    value: Any,
    reducer: Optional[str]
) -> Dict[str, Any]:
    """
    Apply a reduce operation to merge a value into the destination args.
    
    Args:
        dst_args: Target argument dictionary (mutated in place)
        param: Parameter name to write to
        value: Value to merge
        reducer: One of "overwrite", "append_list", "merge_dict", "concat_text", or None (defaults to overwrite)
        
    Returns:
        The mutated dst_args
    """
    if reducer in (None, "overwrite"):
        dst_args[param] = value
    elif reducer == "append_list":
        dst_args.setdefault(param, [])
        dst_args[param].append(value)
    elif reducer == "merge_dict":
        base = dst_args.get(param, {})
        base = base if isinstance(base, dict) else {}
        dst_args[param] = {**base, **(value or {})}
    elif reducer == "concat_text":
        prev = dst_args.get(param, "")
        separator = "\n" if prev else ""
        dst_args[param] = (prev + separator + (value or "")).strip()
    else:
        dst_args[param] = value
    return dst_args


def resolve_bindings(
    bindings: Dict[str, List[Binding]],
    results_by_node: Dict[str, Any],
    dst_args: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Resolve all bindings from an edge, extracting data from source node results
    and applying transformations and reducers.
    
    Args:
        bindings: Edge bindings mapping param names to lists of Binding objects
        results_by_node: Results from all completed nodes, keyed by node id
        dst_args: Target argument dictionary to write resolved values into
        
    Returns:
        The mutated dst_args with resolved binding values
    """
    for param, binding_list in (bindings or {}).items():
        for b in binding_list:
            if b.literal is not None:
                val = b.literal
            else:
                source_result = results_by_node.get(b.from_node)
                val = get_by_path(source_result, b.path)
            
            if b.transform == "text" and val is not None:
                val = str(val)
            elif b.transform == "json" and val is not None:
                import json
                val = json.dumps(val) if not isinstance(val, str) else val
            
            apply_reduce(dst_args, b.alias or param, val, b.reduce)
    
    return dst_args


def merge_activation(
    prev: Activation,
    node_defaults: Dict[str, Any],
    edge: EdgeSpec,
    results_by_node: Dict[str, Any]
) -> Activation:
    """
    Build the activation for a destination node by merging:
    1. Previous activation state (from earlier edges targeting the same node)
    2. Node defaults (prompt and args from the node spec)
    3. Edge data (prompt, bindings resolved from source node results)
    
    Args:
        prev: Previous activation for the destination node (may be empty)
        node_defaults: Default configuration from the destination node spec
        edge: The edge being traversed
        results_by_node: Results from all completed nodes
        
    Returns:
        Merged Activation for the destination node
    """
    out: Activation = {
        "prompt": prev.get("prompt", ""),
        "args": dict(prev.get("args", {}))
    }
    
    # Apply node defaults (only if not already set)
    nd = node_defaults or {}
    ndp = nd.get("prompt")
    if ndp and not out["prompt"]:
        out["prompt"] = ndp
    for k, v in (nd.get("args") or {}).items():
        out["args"].setdefault(k, v)
    
    # Apply edge prompt
    if edge.prompt:
        if edge.merge == "accumulate" and out["prompt"]:
            out["prompt"] = (out["prompt"] + "\n" + edge.prompt).strip()
        else:
            out["prompt"] = edge.prompt
    
    # Resolve edge bindings
    resolve_bindings(edge.bindings, results_by_node, out["args"])
    
    return out
