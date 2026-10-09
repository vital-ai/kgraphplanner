"""
KGraphSerializer must implement LangGraph's typed serializer protocol.

The bug: dumps_typed() returned a KGraphSerialized object instead of a
(type, bytes) tuple. Since the checkpointers moved to LangGraph's blob layout
(v0.1.3), reloading any thread checkpointed with KGraphSerializer failed:
"'KGraphSerialized' object is not subscriptable" (memory) and "cannot unpack
non-iterable KGraphSerialized object" (Redis). It also stored values as plain
JSON, so messages came back as dicts.

What must hold now:
  - dumps_typed/loads_typed round-trip state with its types (messages, tool calls);
  - a checkpointed conversation reloads across turns;
  - values written by the old format still load.
"""
import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from kgraphplanner.agent.kgraph_chat_agent import KGraphChatAgent
from kgraphplanner.checkpointer.kgraph_serialized import KGraphSerialized
from kgraphplanner.checkpointer.kgraph_serializer import KGraphSerializer
from kgraphplanner.checkpointer.kgraphmemory_checkpointer import KGraphMemoryCheckpointer
from kgraphplanner.testing import ScriptedLLM
from kgraphplanner.worker.kgraph_chat_worker import KGraphChatWorker


def test_typed_round_trip_keeps_types():
    serde = KGraphSerializer()
    state = {
        "messages": [HumanMessage(content="hi"),
                     AIMessage(content="", tool_calls=[{"name": "t", "args": {"q": 1}, "id": "c1"}])],
        "work": {"tw": {"tool_history": [[AIMessage(content="x"),
                                          [ToolMessage(content="r", tool_call_id="c1")]]]}},
    }
    type_, payload = serde.dumps_typed(state)
    assert type_.startswith("kgraph+") and isinstance(payload, bytes)
    back = serde.loads_typed((type_, payload))
    assert isinstance(back["messages"][1], AIMessage)
    assert back["messages"][1].tool_calls[0]["args"] == {"q": 1}
    assert isinstance(back["work"]["tw"]["tool_history"][0][1][0], ToolMessage)


def test_legacy_container_still_loads():
    legacy = KGraphSerialized(data=json.dumps({"a": 1}).encode(), original_type="dict")
    assert KGraphSerializer().loads_typed(legacy) == {"a": 1}


async def test_checkpointed_conversation_reloads_across_turns():
    llm = ScriptedLLM(default=["first reply", "second reply"])
    agent = KGraphChatAgent(name="chat", chat_worker=KGraphChatWorker(name="chat", llm=llm),
                            checkpointer=KGraphMemoryCheckpointer(serde=KGraphSerializer()))
    config = {"configurable": {"thread_id": "t1"}}
    await agent.arun([HumanMessage(content="one")], config=config)
    state = await agent.arun([HumanMessage(content="two")], config=config)
    assert [type(m).__name__ for m in state["messages"]] == [
        "HumanMessage", "AIMessage", "HumanMessage", "AIMessage"]
    # The second call saw the reloaded first turn.
    assert "first reply" in [m.content for m in llm.calls[1]["messages"]]
