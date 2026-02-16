"""
Minimal test to verify LangGraph cycle support.

Tests two scenarios:
1. Simple cycle with conditional back-edge (should work)
2. Mixed structural + conditional edges to same node (the pattern we need)
"""
import asyncio
from typing import TypedDict, Dict, Any, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from operator import or_ as merge_dicts


class TestState(TypedDict):
    counter: int
    log: Annotated[Dict[str, Any], merge_dicts]


# ============================================================
# Test 1: Simple cycle with conditional back-edge
# ============================================================
def test_simple_cycle():
    """A → B → A (loop) or B → END (exit). Pure conditional routing."""
    print("=" * 60)
    print("Test 1: Simple cycle with conditional back-edge")
    print("=" * 60)

    graph = StateGraph(TestState)

    def node_a(state):
        c = state["counter"] + 1
        print(f"  node_a: counter {state['counter']} → {c}")
        return {"counter": c, "log": {"a_runs": state.get("log", {}).get("a_runs", 0) + 1}}

    def node_b(state):
        c = state["counter"] + 10
        print(f"  node_b: counter {state['counter']} → {c}")
        return {"counter": c, "log": {"b_runs": state.get("log", {}).get("b_runs", 0) + 1}}

    graph.add_node("a", node_a)
    graph.add_node("b", node_b)

    graph.add_edge(START, "a")
    graph.add_edge("a", "b")
    # Conditional back-edge: loop if counter < 25, else exit
    graph.add_conditional_edges("b", lambda s: "loop" if s["counter"] < 25 else "done",
                                 {"loop": "a", "done": END})

    compiled = graph.compile()
    result = compiled.invoke({"counter": 0, "log": {}})
    print(f"  Result: counter={result['counter']}, log={result['log']}")
    assert result["counter"] >= 25, f"Expected counter >= 25, got {result['counter']}"
    print("  ✅ PASS\n")


# ============================================================
# Test 2: Mixed structural + conditional edges to same node
#   This mimics our target topology:
#   - initial → chat (structural, first entry)
#   - tool → chat (conditional back-edge, loop)
#   - chat → tool (conditional, lookup)
#   - chat → END (conditional, done)
# ============================================================
def test_mixed_edges_to_same_node():
    """
    initial → chat (structural)
    chat → tool | END (conditional)
    tool → chat (conditional back-edge)
    """
    print("=" * 60)
    print("Test 2: Mixed structural + conditional to same node")
    print("=" * 60)

    graph = StateGraph(TestState)

    iteration = {"count": 0}

    def initial_node(state):
        print(f"  initial_node: setting up context")
        return {"counter": 0, "log": {"initial": True, "found": False}}

    def chat_node(state):
        found = state.get("log", {}).get("found", False)
        iteration["count"] += 1
        print(f"  chat_node (iter {iteration['count']}): found={found}")
        if found:
            return {"log": {"chat_action": "continue", "found": True}}
        else:
            return {"log": {"chat_action": "lookup", "found": False}}

    def tool_node(state):
        # Simulate: first call fails, second succeeds
        attempts = state.get("log", {}).get("tool_attempts", 0) + 1
        found = attempts >= 2
        print(f"  tool_node: attempt {attempts}, found={found}")
        return {"log": {"tool_attempts": attempts, "found": found}}

    graph.add_node("initial", initial_node)
    graph.add_node("chat", chat_node)
    graph.add_node("tool", tool_node)

    # Structural edges
    graph.add_edge(START, "initial")
    graph.add_edge("initial", "chat")  # structural first entry

    # Conditional from chat: lookup → tool, continue → END
    graph.add_conditional_edges(
        "chat",
        lambda s: "lookup" if s.get("log", {}).get("chat_action") == "lookup" else "done",
        {"lookup": "tool", "done": END}
    )

    # Conditional back-edge: tool → chat (NOT structural add_edge!)
    graph.add_conditional_edges(
        "tool",
        lambda s: "back_to_chat",
        {"back_to_chat": "chat"}
    )

    compiled = graph.compile()
    result = compiled.invoke({"counter": 0, "log": {}})
    print(f"  Result: log={result['log']}")
    assert result["log"].get("found") == True, "Expected found=True"
    assert result["log"].get("tool_attempts", 0) >= 2, "Expected at least 2 tool attempts"
    print("  ✅ PASS\n")


# ============================================================
# Test 3: DEADLOCK scenario — fan-in with structural edges
#   Both initial and tool use add_edge to chat → deadlock expected
# ============================================================
def test_fanin_deadlock():
    """
    initial → chat (structural)
    tool → chat (structural) ← THIS SHOULD DEADLOCK
    chat → tool | END (conditional)
    """
    print("=" * 60)
    print("Test 3: Fan-in structural (expected deadlock/timeout)")
    print("=" * 60)

    graph = StateGraph(TestState)

    iteration = {"count": 0}

    def initial_node(state):
        print(f"  initial_node: setting up")
        return {"counter": 0, "log": {"initial": True, "found": False}}

    def chat_node(state):
        found = state.get("log", {}).get("found", False)
        iteration["count"] += 1
        print(f"  chat_node (iter {iteration['count']}): found={found}")
        if found:
            return {"log": {"chat_action": "continue", "found": True}}
        else:
            return {"log": {"chat_action": "lookup", "found": False}}

    def tool_node(state):
        attempts = state.get("log", {}).get("tool_attempts", 0) + 1
        found = attempts >= 2
        print(f"  tool_node: attempt {attempts}, found={found}")
        return {"log": {"tool_attempts": attempts, "found": found}}

    graph.add_node("initial", initial_node)
    graph.add_node("chat", chat_node)
    graph.add_node("tool", tool_node)

    # Structural edges
    graph.add_edge(START, "initial")
    graph.add_edge("initial", "chat")   # structural first entry
    graph.add_edge("tool", "chat")       # structural back-edge ← PROBLEM

    # Conditional from chat
    graph.add_conditional_edges(
        "chat",
        lambda s: "lookup" if s.get("log", {}).get("chat_action") == "lookup" else "done",
        {"lookup": "tool", "done": END}
    )

    try:
        compiled = graph.compile()
        # Use asyncio with a timeout to detect deadlock
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError("Deadlock detected!")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(5)  # 5 second timeout

        try:
            result = compiled.invoke({"counter": 0, "log": {}})
            signal.alarm(0)
            # If it completes, LangGraph may handle this differently than expected
            print(f"  Result: log={result['log']}")
            print("  ⚠️  No deadlock — LangGraph may handle fan-in cycles differently")
            print("  (This means structural back-edges might work in newer LangGraph versions)\n")
        except TimeoutError:
            signal.alarm(0)
            print("  ✅ CONFIRMED: Deadlock with structural fan-in back-edge\n")
        except Exception as e:
            signal.alarm(0)
            print(f"  ❌ Error (not deadlock): {type(e).__name__}: {e}\n")

    except Exception as e:
        print(f"  ❌ Graph compilation error: {type(e).__name__}: {e}")
        print("  (LangGraph may reject cycles with structural edges at compile time)\n")


# ============================================================
# Test 4: Intermediate gather node (mimics build_graph pattern)
#   initial_exit → gather → chat_entry
#   tool_exit → gather → chat_entry
#   This is the actual topology build_graph would create.
# ============================================================
def test_gather_node_cycle():
    """
    Mimics build_graph: a gather node sits between sources and destination.
    initial → gather ← tool (fan-in at gather)
    gather → chat → tool → gather (cycle through gather)
    """
    print("=" * 60)
    print("Test 4: Gather-node fan-in with cycle (build_graph pattern)")
    print("=" * 60)

    graph = StateGraph(TestState)

    iteration = {"count": 0}

    def initial_node(state):
        print(f"  initial_node: setting up")
        return {"counter": 0, "log": {"initial": True, "found": False}}

    def gather_node(state):
        print(f"  gather_node: merging activation data")
        return {"log": {"gathered": True}}

    def chat_node(state):
        found = state.get("log", {}).get("found", False)
        iteration["count"] += 1
        print(f"  chat_node (iter {iteration['count']}): found={found}")
        if found:
            return {"log": {"chat_action": "continue", "found": True}}
        else:
            return {"log": {"chat_action": "lookup", "found": False}}

    def tool_node(state):
        attempts = state.get("log", {}).get("tool_attempts", 0) + 1
        found = attempts >= 2
        print(f"  tool_node: attempt {attempts}, found={found}")
        return {"log": {"tool_attempts": attempts, "found": found}}

    graph.add_node("initial", initial_node)
    graph.add_node("gather", gather_node)
    graph.add_node("chat", chat_node)
    graph.add_node("tool", tool_node)

    # Structural edges — both feed into gather (fan-in)
    graph.add_edge(START, "initial")
    graph.add_edge("initial", "gather")  # first entry to gather
    graph.add_edge("tool", "gather")      # back-edge to gather

    # gather → chat (structural)
    graph.add_edge("gather", "chat")

    # Conditional from chat: lookup → tool, continue → END
    graph.add_conditional_edges(
        "chat",
        lambda s: "lookup" if s.get("log", {}).get("chat_action") == "lookup" else "done",
        {"lookup": "tool", "done": END}
    )

    try:
        compiled = graph.compile()

        import signal
        def timeout_handler(signum, frame):
            raise TimeoutError("Deadlock detected!")
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(10)  # 10 second timeout

        try:
            result = compiled.invoke({"counter": 0, "log": {}})
            signal.alarm(0)
            print(f"  Result: log={result['log']}")
            assert result["log"].get("found") == True, "Expected found=True"
            print("  ✅ PASS — gather-node cycle works!\n")
        except TimeoutError:
            signal.alarm(0)
            print("  ❌ DEADLOCK: gather node waits for both predecessors\n")
            print("     → build_graph needs back-edge handling\n")
        except Exception as e:
            signal.alarm(0)
            print(f"  ❌ Error: {type(e).__name__}: {e}\n")

    except Exception as e:
        print(f"  ❌ Compilation error: {type(e).__name__}: {e}\n")


# ============================================================
# Test 5: Separate hop nodes (no gather) with cycle
#   initial → hop_a → chat
#   tool → hop_b → chat (separate hop, both structural to chat)
# ============================================================
def test_separate_hops_cycle():
    """
    Two separate hop nodes each with add_edge to chat.
    This is what build_graph does when there's no fan-in (single edge per dest).
    But what if both point to the same dest?
    """
    print("=" * 60)
    print("Test 5: Separate hop nodes, both structural to chat")
    print("=" * 60)

    graph = StateGraph(TestState)

    iteration = {"count": 0}

    def initial_node(state):
        print(f"  initial_node: setting up")
        return {"counter": 0, "log": {"initial": True, "found": False}}

    def hop_a(state):
        print(f"  hop_a: from initial")
        return {"log": {"hop_a": True}}

    def hop_b(state):
        print(f"  hop_b: from tool (back-edge)")
        return {"log": {"hop_b": True}}

    def chat_node(state):
        found = state.get("log", {}).get("found", False)
        iteration["count"] += 1
        print(f"  chat_node (iter {iteration['count']}): found={found}")
        if found:
            return {"log": {"chat_action": "continue", "found": True}}
        else:
            return {"log": {"chat_action": "lookup", "found": False}}

    def tool_node(state):
        attempts = state.get("log", {}).get("tool_attempts", 0) + 1
        found = attempts >= 2
        print(f"  tool_node: attempt {attempts}, found={found}")
        return {"log": {"tool_attempts": attempts, "found": found}}

    graph.add_node("initial", initial_node)
    graph.add_node("hop_a", hop_a)
    graph.add_node("hop_b", hop_b)
    graph.add_node("chat", chat_node)
    graph.add_node("tool", tool_node)

    graph.add_edge(START, "initial")
    graph.add_edge("initial", "hop_a")
    graph.add_edge("hop_a", "chat")      # structural
    graph.add_edge("tool", "hop_b")
    graph.add_edge("hop_b", "chat")      # structural (back-edge path)

    graph.add_conditional_edges(
        "chat",
        lambda s: "lookup" if s.get("log", {}).get("chat_action") == "lookup" else "done",
        {"lookup": "tool", "done": END}
    )

    try:
        compiled = graph.compile()

        import signal
        def timeout_handler(signum, frame):
            raise TimeoutError("Deadlock detected!")
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(10)

        try:
            result = compiled.invoke({"counter": 0, "log": {}})
            signal.alarm(0)
            print(f"  Result: log={result['log']}")
            assert result["log"].get("found") == True, "Expected found=True"
            print("  ✅ PASS — separate hops with cycle works!\n")
        except TimeoutError:
            signal.alarm(0)
            print("  ❌ DEADLOCK: chat waits for both hop_a and hop_b\n")
        except Exception as e:
            signal.alarm(0)
            print(f"  ❌ Error: {type(e).__name__}: {e}\n")

    except Exception as e:
        print(f"  ❌ Compilation error: {type(e).__name__}: {e}\n")


if __name__ == "__main__":
    print("\nLangGraph Cycle Support Verification")
    print("=" * 60)
    print()

    test_simple_cycle()
    test_mixed_edges_to_same_node()
    test_fanin_deadlock()
    test_gather_node_cycle()
    test_separate_hops_cycle()

    print("All tests complete.")
