"""
Case: Unit tests for graph_hop.py

Tests get_by_path, apply_reduce (all reducer types), resolve_bindings,
and merge_activation without any LLM calls.
Fast, deterministic, no external dependencies.
"""

from __future__ import annotations

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from kgraphplanner.graph.graph_hop import (
    get_by_path, apply_reduce, resolve_bindings, merge_activation
)
from kgraphplanner.graph.exec_graph import EdgeSpec, Binding
from test_scripts_planner.cases.test_result import TestResult


async def run() -> TestResult:
    """Run all graph_hop unit tests."""
    failures = []

    # ==================== get_by_path ====================

    data = {
        "name": "Alice",
        "address": {"city": "NYC", "zip": "10001"},
        "tags": ["ai", "ml", "nlp"],
        "nested": {"list": [{"id": 1}, {"id": 2}]}
    }

    # 1. Root path
    if get_by_path(data, "$") != data:
        failures.append("get_by_path($) should return root")

    # 2. Empty/None paths return root
    for p in (None, "", "$"):
        if get_by_path(data, p) != data:
            failures.append(f"get_by_path({p!r}) should return root")

    # 3. Simple key
    if get_by_path(data, "$.name") != "Alice":
        failures.append("get_by_path($.name) should return 'Alice'")

    # 4. Nested key
    if get_by_path(data, "$.address.city") != "NYC":
        failures.append("get_by_path($.address.city) should return 'NYC'")

    # 5. List index
    if get_by_path(data, "$.tags[1]") != "ml":
        failures.append("get_by_path($.tags[1]) should return 'ml'")

    # 6. Nested list index
    if get_by_path(data, "$.nested.list[0].id") != 1:
        failures.append("get_by_path($.nested.list[0].id) should return 1")

    # 7. Missing key returns None
    if get_by_path(data, "$.missing") is not None:
        failures.append("get_by_path($.missing) should return None")

    # 8. Out-of-bounds index returns None
    if get_by_path(data, "$.tags[99]") is not None:
        failures.append("get_by_path($.tags[99]) should return None")

    # 9. Path into non-dict returns None
    if get_by_path(data, "$.name.foo") is not None:
        failures.append("get_by_path($.name.foo) should return None")

    # 10. Path without $. prefix
    if get_by_path(data, "name") != "Alice":
        failures.append("get_by_path('name') without $ should work")

    print(f"  get_by_path: {10 - sum(1 for f in failures if 'get_by_path' in f)}/10 passed")

    # ==================== apply_reduce ====================

    # 11. overwrite (default)
    args = {}
    apply_reduce(args, "key", "val1", None)
    if args["key"] != "val1":
        failures.append("apply_reduce(None) should overwrite")
    apply_reduce(args, "key", "val2", "overwrite")
    if args["key"] != "val2":
        failures.append("apply_reduce('overwrite') should overwrite")

    # 12. append_list
    args = {}
    apply_reduce(args, "items", "a", "append_list")
    apply_reduce(args, "items", "b", "append_list")
    apply_reduce(args, "items", "c", "append_list")
    if args["items"] != ["a", "b", "c"]:
        failures.append(f"append_list: expected ['a','b','c'], got {args['items']}")

    # 13. merge_dict
    args = {}
    apply_reduce(args, "data", {"x": 1}, "merge_dict")
    apply_reduce(args, "data", {"y": 2}, "merge_dict")
    apply_reduce(args, "data", {"x": 99}, "merge_dict")
    if args["data"] != {"x": 99, "y": 2}:
        failures.append(f"merge_dict: expected {{x:99,y:2}}, got {args['data']}")

    # 14. concat_text
    args = {}
    apply_reduce(args, "text", "Hello", "concat_text")
    apply_reduce(args, "text", "World", "concat_text")
    if args["text"] != "Hello\nWorld":
        failures.append(f"concat_text: expected 'Hello\\nWorld', got {args['text']!r}")

    # 15. concat_text with empty initial
    args = {"text": ""}
    apply_reduce(args, "text", "First", "concat_text")
    if args["text"] != "First":
        failures.append(f"concat_text empty start: expected 'First', got {args['text']!r}")

    # 16. append_list creates list from nothing
    args = {}
    apply_reduce(args, "new_list", 42, "append_list")
    if args["new_list"] != [42]:
        failures.append(f"append_list new key: expected [42], got {args['new_list']}")

    # 17. merge_dict with None value
    args = {}
    apply_reduce(args, "d", None, "merge_dict")
    if args["d"] != {}:
        failures.append(f"merge_dict None: expected {{}}, got {args['d']}")

    reduce_fails = sum(1 for f in failures if 'reduce' in f.lower() or 'concat' in f.lower()
                       or 'merge_dict' in f.lower() or 'append_list' in f.lower())
    print(f"  apply_reduce: {7 - reduce_fails}/7 passed")

    # ==================== resolve_bindings ====================

    results_by_node = {
        "research_0": {"result_text": "Research findings A", "score": 0.95},
        "research_1": {"result_text": "Research findings B", "score": 0.80},
        "start": {"topic": "AI", "companies": ["X", "Y"]},
    }

    # 18. Simple binding from_node + path
    dst = {}
    resolve_bindings(
        {"input": [Binding(from_node="research_0", path="$.result_text")]},
        results_by_node, dst
    )
    if dst.get("input") != "Research findings A":
        failures.append(f"resolve simple: expected 'Research findings A', got {dst.get('input')}")

    # 19. Binding with alias
    dst = {}
    resolve_bindings(
        {"data": [Binding(from_node="research_0", path="$.score", alias="confidence")]},
        results_by_node, dst
    )
    if dst.get("confidence") != 0.95:
        failures.append(f"resolve alias: expected 0.95 at 'confidence', got {dst}")

    # 20. Binding with reduce=append_list
    dst = {}
    resolve_bindings(
        {"items": [
            Binding(from_node="research_0", path="$", reduce="append_list"),
            Binding(from_node="research_1", path="$", reduce="append_list"),
        ]},
        results_by_node, dst
    )
    if len(dst.get("items", [])) != 2:
        failures.append(f"resolve append: expected 2 items, got {dst.get('items')}")

    # 21. Binding with literal value
    dst = {}
    resolve_bindings(
        {"mode": [Binding(literal="summary")]},
        results_by_node, dst
    )
    if dst.get("mode") != "summary":
        failures.append(f"resolve literal: expected 'summary', got {dst.get('mode')}")

    # 22. Binding with transform=text
    dst = {}
    resolve_bindings(
        {"val": [Binding(from_node="research_0", path="$.score", transform="text")]},
        results_by_node, dst
    )
    if dst.get("val") != "0.95":
        failures.append(f"resolve transform=text: expected '0.95', got {dst.get('val')}")

    # 23. Empty bindings is a no-op
    dst = {"existing": "keep"}
    resolve_bindings({}, results_by_node, dst)
    resolve_bindings(None, results_by_node, dst)
    if dst != {"existing": "keep"}:
        failures.append(f"empty bindings should be no-op, got {dst}")

    resolve_fails = sum(1 for f in failures if 'resolve' in f.lower())
    print(f"  resolve_bindings: {6 - resolve_fails}/6 passed")

    # ==================== merge_activation ====================

    # 24. Basic merge with edge prompt and bindings
    edge = EdgeSpec(
        source="research_0", destination="summarizer",
        prompt="Summarize the research",
        bindings={"input": [Binding(from_node="research_0", path="$.result_text")]}
    )
    act = merge_activation(
        {"prompt": "", "args": {}},
        {"prompt": "Default prompt", "args": {"mode": "brief"}},
        edge, results_by_node
    )
    if act["prompt"] != "Summarize the research":
        failures.append(f"merge prompt: expected edge prompt, got {act['prompt']!r}")
    if act["args"].get("input") != "Research findings A":
        failures.append(f"merge binding: expected research text, got {act['args']}")
    if act["args"].get("mode") != "brief":
        failures.append(f"merge defaults: expected mode='brief', got {act['args']}")

    # 25. Node defaults only when prompt not already set
    edge_no_prompt = EdgeSpec(source="s", destination="d")
    act = merge_activation(
        {"prompt": "", "args": {}},
        {"prompt": "From defaults"},
        edge_no_prompt, {}
    )
    if act["prompt"] != "From defaults":
        failures.append(f"merge default prompt: expected 'From defaults', got {act['prompt']!r}")

    # 26. Existing prompt not overridden by defaults
    act = merge_activation(
        {"prompt": "Already set", "args": {}},
        {"prompt": "From defaults"},
        edge_no_prompt, {}
    )
    if act["prompt"] != "Already set":
        failures.append(f"merge keep prompt: expected 'Already set', got {act['prompt']!r}")

    # 27. Accumulate merge mode
    edge_accum = EdgeSpec(source="s", destination="d",
                          prompt="Additional instruction", merge="accumulate")
    act = merge_activation(
        {"prompt": "Base prompt", "args": {}},
        {},
        edge_accum, {}
    )
    if "Base prompt" not in act["prompt"] or "Additional instruction" not in act["prompt"]:
        failures.append(f"merge accumulate: expected both prompts, got {act['prompt']!r}")

    # 28. Node default args fill in missing keys only
    edge_empty = EdgeSpec(source="s", destination="d")
    act = merge_activation(
        {"prompt": "", "args": {"existing": "keep"}},
        {"args": {"existing": "override_attempt", "new_key": "added"}},
        edge_empty, {}
    )
    if act["args"]["existing"] != "keep":
        failures.append(f"merge setdefault: 'existing' should not be overridden")
    if act["args"].get("new_key") != "added":
        failures.append(f"merge setdefault: 'new_key' should be added")

    # 29. None node_defaults handled safely
    act = merge_activation(
        {"prompt": "", "args": {}},
        None,
        edge_empty, {}
    )
    # Should not raise

    # 30. None args in node_defaults handled safely
    act = merge_activation(
        {"prompt": "", "args": {}},
        {"args": None, "prompt": None},
        edge_empty, {}
    )
    # Should not raise

    merge_fails = sum(1 for f in failures if 'merge' in f.lower())
    print(f"  merge_activation: {7 - merge_fails}/7 passed")

    # --- Summary ---
    total_tests = 30
    passed = total_tests - len(failures)
    print(f"\n  Subtests: {passed}/{total_tests} passed")

    if failures:
        for f in failures:
            print(f"  FAIL: {f}")
        assert False, f"{len(failures)} subtests failed: {failures[0]}"

    return TestResult(
        name="Unit: graph_hop",
        passed=True,
        details={"subtests_passed": passed, "subtests_total": total_tests}
    )
