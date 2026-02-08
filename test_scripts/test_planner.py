from __future__ import annotations

from dataclasses import dataclass
from typing import Union, Type, Any, Dict, List, Literal, Optional, TypedDict

import json
from pydantic import BaseModel, Field, ValidationError, ConfigDict

# LangGraph 1.0.8
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.runtime import get_runtime  # Context + stream writer

# LangChain
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage
)
from langchain_core.tools import BaseTool
import re

from dotenv import load_dotenv
import asyncio
from typing_extensions import Annotated
from operator import or_ as merge_dicts  # dict union: left | right
from collections.abc import Mapping
from langchain_core.runnables.graph import MermaidDrawMethod


load_dotenv()

def _safe_id(*parts: str) -> str:
    # allow letters, digits, underscore, dot, dash; replace everything else with "_"
    return re.sub(r"[^A-Za-z0-9_.-]", "_", "__".join(parts))

def _ev_get(ev, key, default=None):
    # Works for dict-like and object-like events
    if isinstance(ev, Mapping):
        return ev.get(key, default)
    return getattr(ev, key, default)

def normalize_event(ev):
    # Try several likely type fields
    typ = (
        _ev_get(ev, "type")
        or _ev_get(ev, "event")
        or ("custom" if _ev_get(ev, "custom") is not None else None)
        or _ev_get(ev, "kind")
        or _ev_get(ev, "op")
        or "unknown"
    )
    node = (
        _ev_get(ev, "node_name")
        or _ev_get(ev, "name")
        or _ev_get(ev, "run_name")
        or _ev_get(ev, "node")
    )
    data = _ev_get(ev, "data")
    if data is None:
        # Fall back to common payload fields
        for k in ("custom", "updates", "messages", "outputs", "inputs", "state", "value"):
            v = _ev_get(ev, k)
            if v is not None:
                data = v
                break
    return typ, node, data

def _require_all_props(schema: dict) -> None:
    # harmless for function_calling, useful if you later flip to JSON mode
    props = schema.get("properties")
    if isinstance(props, dict):
        schema["required"] = list(props.keys())


class Binding(BaseModel):
    from_node: Optional[str] = None
    path: str = "$"
    literal: Optional[Any] = None   # internal only
    transform: Optional[Literal["as_is","text","json"]] = "as_is"
    reduce: Optional[Literal["overwrite","append_list","merge_dict","concat_text"]] = None
    alias: Optional[str] = None

class EdgeSpec(BaseModel):
    source: str
    destination: str
    prompt: Optional[str] = None
    bindings: Dict[str, List[Binding]] = Field(default_factory=dict)
    merge: Literal["edge_overrides","node_defaults_then_edge","accumulate"] = "edge_overrides"

class NodeSpec(BaseModel):
    id: str
    worker: str
    defaults: Dict[str, Any] = Field(default_factory=dict)

class GraphSpec(BaseModel):
    nodes: List[NodeSpec]
    edges: List[EdgeSpec]
    exit_nodes: List[str] = Field(default_factory=list)
    max_parallel: int = Field(3, ge=1, le=8)

def repair_graphspec(g: GraphSpec) -> GraphSpec:
    seen, nodes = set(), []
    for n in g.nodes:
        if n.id in seen:
            continue
        seen.add(n.id)
        nodes.append(n)
    ids = {n.id for n in nodes}
    edges = [e for e in g.edges if e.source in ids and e.destination in ids and e.source != e.destination]
    exits = [x for x in g.exit_nodes if x in ids]
    mp = g.max_parallel if 1 <= g.max_parallel <= 8 else 3
    return GraphSpec(nodes=nodes, edges=edges, exit_nodes=exits, max_parallel=mp)

# ============================================================
# B) JSON-Schema-friendly ProgramSpec (planner output)
#    NO `Any`, no recursive arbitrary unions.
# ============================================================

# Allowed scalar JSON values in planner schemas
JsonScalar = Union[str, int, float, bool, None]
# Modest arg value set we actually need in examples:
#  - scalars
#  - list of strings (for $.companies)
JsonArgValue = Union[JsonScalar, List[str]]
JsonArgs = Dict[str, JsonArgValue]

def _fmt(x: Any, ctx: Dict[str, Any]) -> Any:
    return x.format(**ctx) if isinstance(x, str) else x

def _fmt_deep(obj: Any, ctx: Dict[str, Any]) -> Any:
    if isinstance(obj, dict): return {k: _fmt_deep(v, ctx) for k, v in obj.items()}
    if isinstance(obj, list): return [_fmt_deep(v, ctx) for v in obj]
    return _fmt(obj, ctx)

# ---- ProgramSpec leaf types (JSON-mode safe) ----

class NodeDefaultsT(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: Optional[str] = None
    args: Optional[JsonArgs] = None

class PSNodeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    worker: str
    defaults: Optional[NodeDefaultsT] = None

class BindingPattern(BaseModel):
    model_config = ConfigDict(extra="forbid")
    from_node_tpl: Optional[str] = None
    path: str = "$"
    # IMPORTANT: literal is now limited to JSON scalars to fit JSON-Schema mode
    literal_tpl: Optional[JsonScalar] = None
    transform: Optional[Literal["as_is","text","json"]] = "as_is"
    reduce: Optional[Literal["overwrite","append_list","merge_dict","concat_text"]] = None
    alias: Optional[str] = None

class EdgePattern(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_tpl: str
    destination_tpl: str
    prompt_tpl: Optional[str] = None
    merge: Literal["edge_overrides","node_defaults_then_edge","accumulate"] = "edge_overrides"
    bindings: Dict[str, List[BindingPattern]] = Field(default_factory=dict)

class NodePattern(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id_tpl: str
    worker: str
    defaults_tmpl: Optional[NodeDefaultsT] = None

class ForEach(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_from: Literal["start_args","literal"] = "start_args"
    source_path: str = "$.items"
    literal_items: Optional[List[JsonScalar]] = None
    item_var: str = "item"
    idx_var: str = "idx"

class FanOutBranch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    destination_tpl: str
    prompt_tpl: Optional[str] = None
    bindings: Dict[str, List[BindingPattern]] = Field(default_factory=dict)
    merge: Literal["edge_overrides","node_defaults_then_edge","accumulate"] = "edge_overrides"

class FanOutSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_tpl: str
    branches: List[FanOutBranch]

class FanInSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sources_tpl: List[str]
    destination_tpl: str
    param: str = "items"
    prompt_tpl: Optional[str] = None
    reduce: Literal["append_list","merge_dict","concat_text"] = "append_list"

class TemplateSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "loop"
    for_each: ForEach
    loop_nodes: List[NodePattern]
    loop_edges: List[EdgePattern] = Field(default_factory=list)
    fan_out: List[FanOutSpec] = Field(default_factory=list)
    fan_in:  List[FanInSpec]  = Field(default_factory=list)

class PSEdgeSpec(BaseModel):
    # Static edges for planner output (keep simple; no bindings here)
    model_config = ConfigDict(extra="forbid")
    source: str
    destination: str
    prompt: Optional[str] = None

class ProgramSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    static_nodes: List[PSNodeSpec] = Field(default_factory=list)
    static_edges: List[PSEdgeSpec] = Field(default_factory=list)
    templates: List[TemplateSpec] = Field(default_factory=list)
    exit_nodes: List[str] = Field(default_factory=list)
    max_parallel: int = 3


# ============================================================
# C) ProgramSpec → GraphSpec expander
# ============================================================

def _resolve_path_simple(root: Any, path: str):
    if path in (None, "", "$"): return root
    p = path
    if p.startswith("$"): p = p[1:]
    if p.startswith("."): p = p[1:]
    cur = root
    if not p: return cur
    for part in p.split("."):
        cur = cur.get(part) if isinstance(cur, dict) else None
        if cur is None: return None
    return cur

def _render_bindings_from_patterns(bdict: Dict[str, List[BindingPattern]], ctx: Dict[str, Any]) -> Dict[str, List[Binding]]:
    out: Dict[str, List[Binding]] = {}
    for param, pats in (bdict or {}).items():
        lst: List[Binding] = []
        for bp in pats:
            from_node = _fmt(bp.from_node_tpl, ctx) if bp.from_node_tpl else None
            lit = _fmt(bp.literal_tpl, ctx) if bp.literal_tpl is not None else None
            lst.append(Binding(from_node=from_node, path=_fmt(bp.path, ctx), literal=lit,
                   transform=bp.transform, reduce=bp.reduce, alias=bp.alias))
        out[param] = lst
    return out

def _dump_defaults_tmpl(dt: Optional[NodeDefaultsT], ctx: Dict[str, Any]) -> Dict[str, Any]:
    if not dt:
        return {}
    d = dt.model_dump(exclude_none=True)
    return _fmt_deep(d, ctx)

def _dump_defaults(dt: Optional[NodeDefaultsT]) -> Dict[str, Any]:
    if not dt:
        return {}
    return dt.model_dump(exclude_none=True)

def expand_program_to_graph(program: ProgramSpec, start_seed: Dict[str, Any] | None = None) -> GraphSpec:
    # Convert static planner nodes → internal GraphSpec nodes
    nodes: List[NodeSpec] = [
        NodeSpec(id=n.id, worker=n.worker, defaults=_dump_defaults(n.defaults))
        for n in program.static_nodes
    ]
    # Static edges → internal edges (bindings unused here)
    edges: List[EdgeSpec] = [EdgeSpec(source=e.source, destination=e.destination, prompt=e.prompt) for e in program.static_edges]

    # find start args (from static "start" if present), allow external seed merge
    start_args = next((n.defaults.get("args", {}) for n in nodes if n.id == "start"), {})
    if start_seed:
        start_args = {**start_args, **start_seed}

    for tmpl in program.templates:
        # resolve loop items
        if tmpl.for_each.source_from == "literal":
            items = tmpl.for_each.literal_items or []
        else:
            items = _resolve_path_simple(start_args, tmpl.for_each.source_path) or []

        for idx, item in enumerate(items):
            ctx = {tmpl.for_each.item_var: item, tmpl.for_each.idx_var: idx}

            # nodes
            for np in tmpl.loop_nodes:
                nid = _fmt(np.id_tpl, ctx)            
                nid = _safe_id(nid)  # sanitize LLM-produced ids

                nodes.append(NodeSpec(id=nid, worker=np.worker, defaults=_dump_defaults_tmpl(np.defaults_tmpl, ctx)))

            # edges (non fan)
            for ep in tmpl.loop_edges:
                src = _fmt(ep.source_tpl, ctx)
                dst = _fmt(ep.destination_tpl, ctx)
                edges.append(EdgeSpec(
                    source=src, destination=dst,
                    prompt=_fmt(ep.prompt_tpl, ctx),
                    bindings=_render_bindings_from_patterns(ep.bindings, ctx),
                    merge=ep.merge
                ))

            # fan-out
            for fo in tmpl.fan_out:
                src = _fmt(fo.source_tpl, ctx)
                for br in fo.branches:
                    dst = _fmt(br.destination_tpl, ctx)
                    edges.append(EdgeSpec(
                        source=src, destination=dst,
                        prompt=_fmt(br.prompt_tpl, ctx),
                        bindings=_render_bindings_from_patterns(br.bindings, ctx),
                        merge=br.merge
                    ))

            # fan-in
            for fi in tmpl.fan_in:
                dst = _fmt(fi.destination_tpl, ctx)
                for src_tpl in fi.sources_tpl:
                    src = _fmt(src_tpl, ctx)
                    edges.append(EdgeSpec(
                        source=src, destination=dst,
                        prompt=_fmt(fi.prompt_tpl, ctx) if fi.prompt_tpl else None,
                        bindings={fi.param: [Binding(from_node=src, path="$", reduce=fi.reduce)]}
                    ))

    return repair_graphspec(GraphSpec(nodes=nodes, edges=edges,
                                      exit_nodes=program.exit_nodes, max_parallel=program.max_parallel))


# ============================================================
# D) Binding resolution at hops → Activation
# ============================================================

class Activation(TypedDict, total=False):
    prompt: str
    args: Dict[str, Any]

def get_by_path(obj, path: str):
    if path in (None, "", "$"): return obj
    cur = obj
    p = path
    if p.startswith("$"): p = p[1:]
    if p.startswith("."): p = p[1:]
    for part in p.split("."):
        if "[" in part and part.endswith("]"):
            key, idx = part[:-1].split("[")
            cur = cur.get(key, None) if isinstance(cur, dict) else None
            if cur is None or not isinstance(cur, list): return None
            try: cur = cur[int(idx)]
            except Exception: return None
        else:
            cur = cur.get(part, None) if isinstance(cur, dict) else None
        if cur is None: return None
    return cur

def apply_reduce(dst_args: Dict[str, Any], param: str, value: Any, reducer: Optional[str]) -> Dict[str, Any]:
    if reducer in (None, "overwrite"):
        dst_args[param] = value
    elif reducer == "append_list":
        dst_args.setdefault(param, []); dst_args[param].append(value)
    elif reducer == "merge_dict":
        base = dst_args.get(param, {}); base = base if isinstance(base, dict) else {}
        dst_args[param] = {**base, **(value or {})}
    elif reducer == "concat_text":
        prev = dst_args.get(param, ""); dst_args[param] = (prev + ("\n" if prev else "") + (value or "")).strip()
    else:
        dst_args[param] = value
    return dst_args

def merge_activation(prev: Activation, node_defaults: Dict[str, Any],
                     edge: EdgeSpec, results_by_node: Dict[str, Any]) -> Activation:
    out: Activation = {"prompt": prev.get("prompt",""), "args": dict(prev.get("args", {}))}
    ndp = node_defaults.get("prompt")
    if ndp and not out["prompt"]: out["prompt"] = ndp
    for k,v in node_defaults.get("args", {}).items(): out["args"].setdefault(k, v)
    if edge.prompt:
        out["prompt"] = (out["prompt"] + ("\n" if out["prompt"] else "") + edge.prompt).strip() \
                        if edge.merge == "accumulate" else edge.prompt
    for param, binds in (edge.bindings or {}).items():
        for b in binds:
            val = b.literal if b.literal is not None else get_by_path(results_by_node.get(b.from_node), b.path)
            if b.transform == "text" and val is not None: val = str(val)
            apply_reduce(out["args"], b.alias or param, val, b.reduce)
    return out


# ============================================================
# E) Tools + Worker (subgraph per occurrence). JSON-mode friendly.
# ============================================================

class WebSearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str
    top_k: int = Field(5, ge=1, le=10)

class WebSearch(BaseTool):
    # Pydantic v2-friendly overrides
    name: str = "web_search"
    description: str = "Stub web search."
    args_schema: Type[WebSearchArgs] = WebSearchArgs

    def _run(self, query: str, top_k: int = 5) -> str:
        hits = [{"title": f"{query} (example)", "url": "https://example.com"} for _ in range(min(3, top_k))]
        return json.dumps({"query": query, "hits": hits})

    async def _arun(self, query: str, top_k: int = 5) -> str:
        return self._run(query, top_k)

@dataclass
class Worker:
    name: str
    llm: ChatOpenAI
    tools: Dict[str, BaseTool]
    system_directive: str
    required_inputs: List[str] = None
    max_iters: int = 6

    def _make_decision_model(self) -> Type[BaseModel]:
        # Build a model whose tool_name is an enum of *this* worker's tools
        # If no tools, keep a dummy value to discourage 'tool'
        tool_keys = tuple(self.tools.keys())
        ToolNameLit = Literal[tool_keys] if tool_keys else Literal["__no_tools__"]

        class WorkerDecision(BaseModel):
            model_config = ConfigDict(extra="forbid", json_schema_extra=_require_all_props)
            type: Literal["tool","final"]
            tool_name: Optional[ToolNameLit] = None
            arguments: Dict[str, JsonArgValue] = Field(default_factory=dict)
            answer: Optional[str] = None

        return WorkerDecision

    def build_subgraph(self, b: StateGraph, occ_id: str) -> tuple[str, str]:
        eval_id = _safe_id(occ_id, "eval")
        tool_id = _safe_id(occ_id, "tool")
        fin_id  = _safe_id(occ_id, "finalize")

        def _get_slot(s: ExecState) -> Dict[str, Any]:
            work = dict(s.get("work", {}))
            slot = dict(work.get(occ_id, {}))
            slot.setdefault("iters", 0)
            slot.setdefault("messages", [])
            work[occ_id] = slot
            s["work"] = work
            return slot

        def _push_tool_msg(slot: Dict[str, Any], tool_name: str, payload: Any):
            msgs: List[BaseMessage] = slot.get("messages", [])
            # switch when using real tools
            # msgs.append(ToolMessage(content=json.dumps(payload)[:2000], name=tool_name))
            text = json.dumps(payload, ensure_ascii=False)[:2000]
            msgs.append(SystemMessage(content=f"[TOOL {tool_name}] {text}"))
            slot["messages"] = msgs[-6:]

        async def eval_node(s: ExecState) -> ExecState:
            runtime = get_runtime()
            writer = runtime.stream_writer

            activation_map = dict(s.get("activation", {}))
            act = activation_map.get(occ_id, {"prompt":"","args":{}})

            slot = _get_slot(s)
            iters = slot["iters"]

            messages: List[BaseMessage] = [SystemMessage(content=self.system_directive)]
            if act.get("prompt"): messages.append(SystemMessage(content=f"Task instructions: {act['prompt']}"))
            messages.append(SystemMessage(content=f"Task args: {act.get('args', {})}"))
            messages.append(SystemMessage(content=f"Available tools: {list(self.tools.keys())}"))
            messages.extend(slot.get("messages", []))

            DecisionModel = self._make_decision_model()

            dec_llm = self.llm.bind(streaming=False, max_retries=3, timeout=60) \
                .with_structured_output(DecisionModel, method="function_calling")
            
            writer({"phase":"worker_eval","node": occ_id, "worker": self.name, "iter": iters})
            
            try:
                dec = await dec_llm.ainvoke(messages)
            except Exception as e:
                # Surface as a breadcrumb and force a safe exit
                writer({"phase":"llm_error", "node": occ_id, "worker": self.name, "error": str(e)})
                msgs = slot.get("messages", [])
                msgs.append(SystemMessage(content=f"[LLM ERROR] {e}"))
                slot["messages"] = msgs[-6:]
                slot["last_decision"] = {"type":"final", "answer":"(auto-final after transient LLM error)"}
                return s

            slot["last_decision"] = dec.model_dump()
            slot["iters"] = iters + 1

            if slot["iters"] > self.max_iters:
                errs = dict(s.get("errors", {}))
                errs[occ_id] = f"Max iterations exceeded in {occ_id}"
                s["errors"] = errs
                return {**s}

            writer({
                "phase":"worker_decision",
                "node": occ_id,
                "worker": self.name,
                "decision": slot["last_decision"]
            })

            return s

        async def tool_node(s: ExecState) -> ExecState:
            runtime = get_runtime()
            writer = runtime.stream_writer
            slot = _get_slot(s)
            dec = slot.get("last_decision", {}) or {}
            tool_name = (dec.get("tool_name") or "")
            planned_args = (dec.get("arguments") or {})

            tool = self.tools.get(tool_name)
            
            if tool is None:
                _push_tool_msg(slot, tool_name or "unknown_tool",
                   {"error": "UNKNOWN_TOOL",
                    "available_tools": list(self.tools.keys())})
                # Add a corrective hint for the next eval pass
                msgs: List[BaseMessage] = slot.get("messages", [])
                msgs.append(SystemMessage(
                    content=f"Invalid tool '{tool_name}'. You must choose one of {list(self.tools.keys())} "
                            f"or return type='final'."))
                slot["messages"] = msgs[-6:]
                return s
            
            
            
            activation_map = dict(s.get("activation") or {})
            act = (activation_map.get(occ_id) or {})
            act_args = dict(act.get("args") or {})
            args = self._coerce_tool_args(tool, planned_args, act_args)
            
            try:
                writer({"phase":"tool_start","node": occ_id, "worker": self.name, "tool": tool_name})
                out = await tool.ainvoke(args)
                writer({"phase":"tool_done","node": occ_id, "worker": self.name, "tool": tool_name})
                slot["last_out"] = out
                _push_tool_msg(slot, tool_name, out)
            except Exception as e:
                slot["last_out"] = {"TOOL_ERROR": str(e)}
                _push_tool_msg(slot, tool_name, slot["last_out"])
                writer({"phase":"tool_error","node": occ_id, "worker": self.name, "tool": tool_name, "error": str(e)})
            return s

        async def finalize_node(s: ExecState) -> ExecState:
            runtime = get_runtime()
            writer = runtime.stream_writer

            res = dict(s.get("results", {}))
            errs = dict(s.get("errors", {}))
            activation_map = dict(s.get("activation", {}))

            slot = _get_slot(s)
            dec = slot.get("last_decision", {}) or {}
            answer = dec.get("answer") or ""
            payload = {"result_text": answer, "args_used": (activation_map.get(occ_id) or {}).get("args", {})}
            res[occ_id] = payload

            writer({"phase":"task_done","node": occ_id, "worker": self.name})

            work = dict(s.get("work", {})); work.pop(occ_id, None)
            activation_map.pop(occ_id, None)

            return {"results": res, "errors": errs, "activation": activation_map, "work": work}

        b.add_node(eval_id, eval_node)
        b.add_node(tool_id, tool_node)
        b.add_node(fin_id,  finalize_node)

        def _route(s: ExecState) -> str:
            slot = (s.get("work") or {}).get(occ_id, {})
            dec = slot.get("last_decision") or {}
            return "TOOL" if (dec.get("type") == "tool") else "FINAL"

        b.add_conditional_edges(eval_id, _route, {"TOOL": tool_id, "FINAL": fin_id})
        b.add_edge(tool_id, eval_id)
        return eval_id, fin_id
    
    def _coerce_tool_args(
        self,
        tool: BaseTool,
        planned: dict,
        act_args: dict,
    ) -> dict:
        """Fill in required args from activation or defaults so tools don't 422."""
        args = dict(planned or {})
        schema = getattr(tool, "args_schema", None)

        # Prefer explicit values from the plan
        if schema and hasattr(schema, "model_fields"):
            for name, field in schema.model_fields.items():
                if name in args:
                    continue
                # Heuristics for common names
                if name == "query":
                    # Build a sensible query from the task args
                    candidate = (
                        act_args.get("query")
                        or act_args.get("company")
                        or act_args.get("topic")
                    )
                    if candidate:
                        args["query"] = str(candidate)
                        continue
                # Use the field default if present
                if field.default is not None:
                    args[name] = field.default

        return args


# ============================================================
# F) Planner (JSON-Schema mode) → ProgramSpec → GraphSpec
# ============================================================

class Planner:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def plan(self, s: dict) -> dict:
        # JSON-Schema mode (default). ProgramSpec and nested models are JSON-mode safe.
        model = self.llm.with_structured_output(ProgramSpec, method="function_calling")

        sys = SystemMessage(content="""
You are a planning compiler. Return a compact ProgramSpec (JSON) to be expanded into a GraphSpec.

Rules:
1) Use a static node "start" with initial literals under start.defaults.args (e.g., {"companies":[...]}).
2) Use TemplateSpec.for_each to iterate (e.g., over $.companies). Provide item_var and idx_var.
3) Prefer the sugar fields:
   - fan_out: one source → multiple destinations (each branch has its own bindings/prompt).
   - fan_in: many sources → one destination with a reducer (append_list/merge_dict/concat_text).
   Use loop_edges only for non-fan edges (e.g., start → first step).
4) Do NOT enumerate parallel edges in loop_edges; use fan_out/fan_in instead.
5) Workers referenced will exist in the runtime registry (e.g., research_worker, analyst_a, analyst_b, aggregator).
6) Keep it succinct; never emit an already-expanded graph. No duplicates.

Return ONLY a valid ProgramSpec JSON object.
""".strip())

        example = AIMessage(content="""
{
  "static_nodes": [
    {"id":"start","worker":"start","defaults":{"args":{"companies":["OpenAI","Anthropic","Vital.ai","Cohere","Mistral"]}}},
    {"id":"aggregate","worker":"aggregator"},
    {"id":"end","worker":"end"}
  ],
  "static_edges": [{"source":"aggregate","destination":"end"}],
  "templates": [
    {
      "name": "per_company",
      "for_each": {"source_from":"start_args","source_path":"$.companies","item_var":"company","idx_var":"idx"},
      "loop_nodes": [
        {"id_tpl":"research_{idx}",  "worker":"research_worker",
         "defaults_tmpl":{"args":{"topic":"{company}"}}},
        {"id_tpl":"analysis_a_{idx}","worker":"analyst_a"},
        {"id_tpl":"analysis_b_{idx}","worker":"analyst_b"}
      ],
      "loop_edges": [
        {"source_tpl":"start","destination_tpl":"research_{idx}",
         "prompt_tpl":"Research the company",
         "bindings":{"company":[{"from_node_tpl":"start","path":"$.companies[{idx}]","reduce":"overwrite"}]}}
      ],
      "fan_out": [
        {"source_tpl":"research_{idx}",
         "branches":[
           {"destination_tpl":"analysis_a_{idx}",
            "prompt_tpl":"Analyze track A",
            "bindings":{"inputA":[{"from_node_tpl":"research_{idx}","path":"$"}]}},
           {"destination_tpl":"analysis_b_{idx}",
            "prompt_tpl":"Analyze track B",
            "bindings":{"inputB":[{"from_node_tpl":"research_{idx}","path":"$"}]}}
         ]}
      ],
      "fan_in": [
        {"sources_tpl":["analysis_a_{idx}","analysis_b_{idx}"],
         "destination_tpl":"aggregate",
         "param":"analyses",
         "reduce":"append_list",
         "prompt_tpl":"Include this analysis"}
      ]
    }
  ],
  "exit_nodes": ["end"],
  "max_parallel": 3
}
""".strip())

        msgs = [sys, example, *s.get("messages", [])]
        try:
            program = model.invoke(msgs)
        except ValidationError:
            # JSON-mode-safe fallback
            program = ProgramSpec(
                static_nodes=[
                    PSNodeSpec(id="start", worker="start", defaults=NodeDefaultsT(args={"companies":["A","B"]})),
                    PSNodeSpec(id="aggregate", worker="aggregator"),
                    PSNodeSpec(id="end", worker="end"),
                ],
                static_edges=[PSEdgeSpec(source="aggregate", destination="end")],
                templates=[
                    TemplateSpec(
                        for_each=ForEach(source_from="start_args", source_path="$.companies", item_var="company", idx_var="idx"),
                        loop_nodes=[
                            NodePattern(id_tpl="research_{idx}",  worker="research_worker",
                                        defaults_tmpl=NodeDefaultsT(args={"topic":"{company}"})),
                            NodePattern(id_tpl="analysis_a_{idx}", worker="analyst_a"),
                            NodePattern(id_tpl="analysis_b_{idx}", worker="analyst_b"),
                        ],
                        loop_edges=[
                            EdgePattern(
                                source_tpl="start", destination_tpl="research_{idx}",
                                prompt_tpl="Research the company",
                                bindings={"company":[BindingPattern(from_node_tpl="start", path="$.companies[{idx}]")]}
                            )
                        ],
                        fan_out=[
                            FanOutSpec(
                                source_tpl="research_{idx}",
                                branches=[
                                    FanOutBranch(
                                        destination_tpl="analysis_a_{idx}",
                                        prompt_tpl="Analyze track A",
                                        bindings={"inputA":[BindingPattern(from_node_tpl="research_{idx}", path="$")]}
                                    ),
                                    FanOutBranch(
                                        destination_tpl="analysis_b_{idx}",
                                        prompt_tpl="Analyze track B",
                                        bindings={"inputB":[BindingPattern(from_node_tpl="research_{idx}", path="$")]}
                                    )
                                ]
                            )
                        ],
                        fan_in=[
                            FanInSpec(
                                sources_tpl=["analysis_a_{idx}","analysis_b_{idx}"],
                                destination_tpl="aggregate",
                                param="analyses",
                                reduce="append_list",
                                prompt_tpl="Include this analysis"
                            )
                        ]
                    )
                ],
                exit_nodes=["end"], max_parallel=3
            )

        graph = expand_program_to_graph(program)
        return {**s, "graph_out": graph.model_dump()}

def build_planning_graph(planner: Planner):
    class PlanState(TypedDict, total=False):
        messages: List[BaseMessage]
        graph_out: Dict[str, Any]
    g = StateGraph(PlanState)
    g.add_node("planner.plan", planner.plan)
    g.add_edge(START, "planner.plan")
    g.add_edge("planner.plan", END)
    return g.compile(checkpointer=MemorySaver())


# ============================================================
# G) Execution: splice worker subgraphs per occurrence + hops
# ============================================================

class ExecState(TypedDict, total=False):
    # allow concurrent writes; values are merged like: left | right
    results:    Annotated[Dict[str, Any], merge_dicts]
    errors:     Annotated[Dict[str, str], merge_dicts]
    activation: Annotated[Dict[str, Activation], merge_dicts]
    work:       Annotated[Dict[str, Any], merge_dicts]
    # keep default (last-value) for scalar routing
    # route: str

def build_exec_graph_from_graphspec(spec: GraphSpec, registry: Dict[str, Worker]):
    b = StateGraph(ExecState)
    by_id = {n.id: n for n in spec.nodes}

    def entry(s: ExecState) -> ExecState:
        results = dict(s.get("results", {}))
        start = next((n for n in spec.nodes if n.id == "start"), None)
        if start and "start" not in results:
            results["start"] = start.defaults.get("args", {})
        return {"results": results, "errors": s.get("errors", {}), "activation": {}, "work": {}}
    b.add_node("__entry__", entry)
    b.add_edge(START, "__entry__")

    entry_for: Dict[str, str] = {}
    exit_for: Dict[str, str]  = {}
    for n in spec.nodes:
        if n.id in ("start","end"):
            continue
        worker = registry.get(n.worker)
        if not worker:
            raise KeyError(f"Worker '{n.worker}' not in registry")
        entry_label, exit_label = worker.build_subgraph(b, n.id)
        entry_for[n.id] = entry_label
        exit_for[n.id]  = exit_label

    def add_hop(src_label: str, edge: EdgeSpec, dest_entry: str):
        dest_node = by_id[edge.destination]
        def hop(s: ExecState) -> ExecState:
            activation_map = dict(s.get("activation", {}))
            prev_act = activation_map.get(edge.destination, {"prompt":"", "args":{}})
            act = merge_activation(prev_act, dest_node.defaults, edge, s.get("results", {}))
            activation_map[edge.destination] = act
            return {"results": s.get("results", {}), "errors": s.get("errors", {}),
                    "activation": activation_map, "work": s.get("work", {})}
        
        hop_id = _safe_id("__hop__", edge.source, "to", edge.destination)

        b.add_node(hop_id, hop)
        b.add_edge(src_label, hop_id)
        b.add_edge(hop_id, dest_entry)

    for e in spec.edges:
        if e.source == "start" and e.destination in entry_for:
            add_hop("__entry__", e, entry_for[e.destination])

    incoming = {n.id: 0 for n in spec.nodes}
    for e in spec.edges:
        incoming[e.destination] = incoming.get(e.destination, 0) + 1
    if not any(e.source == "start" for e in spec.edges):
        for n in spec.nodes:
            if n.id in ("start","end"): continue
            if incoming.get(n.id, 0) == 0:
                add_hop("__entry__", EdgeSpec(source="__entry__", destination=n.id), entry_for[n.id])

    for e in spec.edges:
        if e.source in exit_for and e.destination in entry_for:
            add_hop(exit_for[e.source], e, entry_for[e.destination])

    for x in spec.exit_nodes:
        if x in exit_for:
            b.add_edge(exit_for[x], END)

    for e in spec.edges:
        if e.destination == "end" and e.source in exit_for:
            b.add_edge(exit_for[e.source], END)

    return b.compile(checkpointer=MemorySaver())


# ============================================================
# H) Optional: validate bindings vs worker-declared inputs
# ============================================================

def validate_bindings_or_throw(spec: GraphSpec, registry: Dict[str, Worker]):
    provided: Dict[str, set] = {}
    for e in spec.edges:
        dst = e.destination
        provided.setdefault(dst, set())
        for param in (e.bindings or {}).keys():
            provided[dst].add(param)

    node_by_id = {n.id: n for n in spec.nodes}
    for nid, node in node_by_id.items():
        if node.worker in ("start","end"): continue
        req = set(registry.get(node.worker, Worker(node.worker, None, {}, "", [])).required_inputs or [])
        defaults = set(node.defaults.get("args", {}).keys())
        missing = req - defaults - provided.get(nid, set())
        if missing:
            raise ValueError(f"Node {nid} ({node.worker}) missing required params: {sorted(missing)}")


# ============================================================
# I) Demo wiring
# ============================================================

def demo():
    model = "gpt-4o-mini"
    plan_llm = ChatOpenAI(model=model, temperature=0)
    generic_llm = ChatOpenAI(model=model, temperature=0)

    web = WebSearch()

    research = Worker("research_worker", generic_llm, {"web_search": web},
                      "Research the company using tools if needed; then finalize.", ["company"])
    analyst_a = Worker("analyst_a", generic_llm, {},
                       "Analyze input A; produce concise, structured notes; then finalize.", ["inputA"])
    analyst_b = Worker("analyst_b", generic_llm, {},
                       "Analyze input B; produce concise, structured notes; then finalize.", ["inputB"])
    aggregator = Worker("aggregator", generic_llm, {},
                        "Aggregate a list of analyses into a single summary; then finalize.", ["analyses"])

    registry = {w.name: w for w in [research, analyst_a, analyst_b, aggregator]}

    # PLAN (JSON-Schema mode)
    planner = Planner(plan_llm)
    pgraph = build_planning_graph(planner)

    image_bytes = pgraph.get_graph().draw_mermaid_png(
        draw_method=MermaidDrawMethod.PYPPETEER
    )

    with open("planner_graph.png", "wb") as f:
        f.write(image_bytes)
    
    req = [HumanMessage(content="Research 5 AI companies, analyze via two tracks per company, and produce a combined report.")]
    p_out = pgraph.invoke({"messages": req},
                          context={}, durability="async",
                          config={"configurable":{"thread_id":"plan-1"}})
    spec = GraphSpec(**p_out["graph_out"])
    spec = repair_graphspec(spec)

    validate_bindings_or_throw(spec, registry)

    # EXECUTE
    exec_graph = build_exec_graph_from_graphspec(spec, registry)


    image_bytes = exec_graph.get_graph().draw_mermaid_png(
        draw_method=MermaidDrawMethod.PYPPETEER
    )

    with open("execution_graph.png", "wb") as f:
        f.write(image_bytes)

    async def run_all(exec_graph, registry):
        print("=== STREAM (events) ===")
        async for ev in exec_graph.astream_events(
            {"results": {}, "errors": {}, "activation": {}, "work": {}},
            context={"workers": registry},
            version="v2",
            subgraphs=True,
            include_types=["custom", "node_state", "chat_model"],
            durability="async",
            config={
                "recursion_limit": 300,                 # give yourself headroom
                "configurable": {"thread_id": "exec-1"}
            }
        ):
            typ, node, data = normalize_event(ev)       # your normalizer
            print(typ, node, data)

        final = await exec_graph.ainvoke(
            {"results": {}, "errors": {}, "activation": {}, "work": {}},
            context={"workers": registry},
            durability="async",
            config={
                "recursion_limit": 300,
                "configurable": {"thread_id": "exec-1"}
            }
        )
        print("FINAL:", final.get("errors"), list((final.get("results") or {}).keys()))
        return final

    asyncio.run(run_all(exec_graph, registry))

if __name__ == "__main__":
    demo()
