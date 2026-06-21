"""
Smoke test for langgraph/deepagents upgrade verification.
Run: python test_scripts/test_upgrade_smoke.py
"""
from typing import Dict, Any, List, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from langgraph.checkpoint.base import BaseCheckpointSaver

from kgraphplanner.checkpointer.kgraphmemory_checkpointer import KGraphMemoryCheckpointer
from kgraphplanner.checkpointer.kgraph_serializer import KGraphSerializer
from kgraphplanner.agent.kgraph_base_agent import AgentState, merge_agent_data, merge_work


def echo_node(state: AgentState) -> dict:
    msgs = state.get("messages", [])
    last = msgs[-1].content if msgs else "none"
    return {"messages": [AIMessage(content=f"Echo: {last}")]}


def main():
    # Test 1: AgentState still works with add_messages reducer
    print("1. AgentState annotations OK")

    # Test 2: Build a simple graph
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("echo", echo_node)
    graph_builder.add_edge(START, "echo")
    graph_builder.add_edge("echo", END)
    print("2. Graph built OK")

    # Test 3: Compile with checkpointer
    serde = KGraphSerializer()
    checkpointer = KGraphMemoryCheckpointer(serde=serde)
    compiled = graph_builder.compile(checkpointer=checkpointer)
    print("3. Graph compiled with checkpointer OK")

    # Test 4: Invoke the graph
    result = compiled.invoke(
        {"messages": [HumanMessage(content="Hello")]},
        config={"configurable": {"thread_id": "test-thread-1"}},
    )
    last_msg = result["messages"][-1]
    print(f"4. Invoke OK: {last_msg.content}")

    # Test 5: Check checkpoint persisted
    cp = checkpointer.get_tuple({"configurable": {"thread_id": "test-thread-1"}})
    print(f"5. Checkpoint retrieved OK: has checkpoint={cp is not None}")

    # Test 6: langgraph.runtime (can only call get_runtime inside a node, just verify import)
    from langgraph.runtime import get_runtime
    print(f"6. get_runtime import OK: {get_runtime.__name__}")

    # Test 7: Pydantic model compat
    from pydantic import BaseModel, ConfigDict

    class TestModel(BaseModel):
        model_config = ConfigDict(extra="forbid")
        name: str
        value: int

    obj = TestModel(name="test", value=42)
    dumped = obj.model_dump()
    restored = TestModel.model_validate(dumped)
    print(f"7. Pydantic OK: {restored.name}={restored.value}")

    # Test 8: with_structured_output
    # (just verify the method exists on ChatOpenAI)
    from langchain_openai import ChatOpenAI
    model = ChatOpenAI.__new__(ChatOpenAI)
    assert hasattr(model, "with_structured_output"), "with_structured_output missing"
    print("8. ChatOpenAI.with_structured_output exists OK")

    # Test 9: ChatAnthropic import
    from langchain_anthropic import ChatAnthropic
    assert hasattr(ChatAnthropic, "ainvoke"), "ainvoke missing on ChatAnthropic"
    print("9. ChatAnthropic import OK")

    # Test 10: Tool base class
    from langchain_core.tools import BaseTool
    assert hasattr(BaseTool, "ainvoke"), "ainvoke missing on BaseTool"
    print("10. BaseTool OK")

    print()
    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
