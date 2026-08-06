"""
Regression tests for the LangGraph checkpoint contract.

The bug these cover: under ``create_deep_agent``, ``messages`` and ``files`` are
``DeltaChannel``s, not ``BinaryOperatorAggregate``. LangGraph deliberately omits
delta channels from ``channel_values`` between snapshots and reconstructs them by
walking the parent chain via ``BaseCheckpointSaver.get_delta_channel_history``.
A checkpointer that stores checkpoints verbatim and never records a parent link
returns a thread with **zero messages** — silently, with no error, and only for
delta channels, so ordinary ``add_messages`` graphs keep passing.

Each test therefore asserts against ``InMemorySaver`` as the reference
implementation rather than against a hardcoded expectation.
"""
import uuid
from typing import Annotated, TypedDict

import pytest
from langchain_core.messages import AIMessage
from langgraph.channels.delta import DeltaChannel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import _messages_delta_reducer, add_messages

from kgraphplanner.checkpointer.kgraphmemory_checkpointer import KGraphMemoryCheckpointer
from kgraphplanner.checkpointer.kgraphredis_checkpointer import KGraphRedisCheckpointer

REDIS_URL = "redis://localhost:6381"


class DeltaState(TypedDict):
    # The channel type create_deep_agent uses for `messages`.
    messages: Annotated[list, DeltaChannel(_messages_delta_reducer)]


class PlainState(TypedDict):
    messages: Annotated[list, add_messages]


def _build(state_type, saver):
    graph = StateGraph(state_type)
    graph.add_node("respond", lambda s: {"messages": [AIMessage(content="reply")]})
    graph.add_edge(START, "respond")
    graph.add_edge("respond", END)
    return graph.compile(checkpointer=saver)


def _two_turns(state_type, saver, thread_id):
    """Run two turns and return the message count visible after each."""
    app = _build(state_type, saver)
    config = {"configurable": {"thread_id": thread_id}}
    app.invoke({"messages": [("user", "first")]}, config)
    after_turn_1 = len(app.get_state(config).values["messages"])
    app.invoke({"messages": [("user", "second")]}, config)
    after_turn_2 = len(app.get_state(config).values["messages"])
    return after_turn_1, after_turn_2


def _redis_saver():
    try:
        saver = KGraphRedisCheckpointer(redis_url=REDIS_URL, ttl=60)
        saver._get_sync_client().ping()
        return saver
    except Exception as exc:  # noqa: BLE001 - any connection failure means skip
        pytest.skip(f"Redis unavailable at {REDIS_URL}: {exc}")


@pytest.fixture(params=["memory", "redis"])
def saver(request):
    if request.param == "memory":
        return KGraphMemoryCheckpointer()
    return _redis_saver()


@pytest.fixture
def thread_id():
    return f"test-{uuid.uuid4()}"


# ----------------------------------------------------------------------
# The regression itself
# ----------------------------------------------------------------------


def test_delta_channel_state_survives_across_turns(saver, thread_id):
    """A DeltaChannel must accumulate across turns, matching InMemorySaver.

    Before the fix this returned (0, 0) — the messages were never recoverable.
    """
    reference = _two_turns(DeltaState, InMemorySaver(), thread_id)
    assert reference == (2, 4), "reference implementation changed; update this test"
    assert _two_turns(DeltaState, saver, thread_id) == reference


def test_plain_reducer_channel_still_works(saver, thread_id):
    """Non-delta channels must be unaffected by the delta handling."""
    assert _two_turns(PlainState, saver, thread_id) == _two_turns(
        PlainState, InMemorySaver(), f"{thread_id}-ref"
    )


def test_delta_channel_recalls_content_not_just_count(saver, thread_id):
    app = _build(DeltaState, saver)
    config = {"configurable": {"thread_id": thread_id}}
    app.invoke({"messages": [("user", "remember the number 41")]}, config)
    app.invoke({"messages": [("user", "and 42")]}, config)

    contents = [m.content for m in app.get_state(config).values["messages"]]
    assert "remember the number 41" in contents
    assert "and 42" in contents


@pytest.mark.asyncio
async def test_delta_channel_survives_via_async_api(saver, thread_id):
    """aget_state is the API LangGraph actually resumes from."""
    app = _build(DeltaState, saver)
    config = {"configurable": {"thread_id": thread_id}}
    await app.ainvoke({"messages": [("user", "first")]}, config)
    await app.ainvoke({"messages": [("user", "second")]}, config)

    state = await app.aget_state(config)
    assert len(state.values["messages"]) == 4


# ----------------------------------------------------------------------
# The contract violations that caused it
# ----------------------------------------------------------------------


def test_put_returns_config_identifying_the_checkpoint(saver, thread_id):
    """put() must return a config naming the checkpoint it wrote.

    LangGraph threads this forward as the *next* put's parent; returning the
    input config unchanged severs the parent chain.
    """
    _build(DeltaState, saver).invoke(
        {"messages": [("user", "hi")]}, {"configurable": {"thread_id": thread_id}}
    )
    tup = saver.get_tuple({"configurable": {"thread_id": thread_id}})
    assert tup.config["configurable"]["checkpoint_id"] == tup.checkpoint["id"]


def test_get_tuple_returns_config_identifying_the_checkpoint(saver, thread_id):
    """A latest-checkpoint lookup must report *which* checkpoint it resolved to.

    Echoing the caller's config leaves the resume path without a checkpoint_id,
    so the next turn's first checkpoint records no parent and the chain snaps at
    the turn boundary.
    """
    _build(DeltaState, saver).invoke(
        {"messages": [("user", "hi")]}, {"configurable": {"thread_id": thread_id}}
    )
    tup = saver.get_tuple({"configurable": {"thread_id": thread_id}})  # no checkpoint_id
    assert tup.config["configurable"].get("checkpoint_id") is not None


def test_checkpoints_form_an_unbroken_parent_chain(saver, thread_id):
    """Every checkpoint but the first must link to its parent, across turns."""
    app = _build(DeltaState, saver)
    config = {"configurable": {"thread_id": thread_id}}
    app.invoke({"messages": [("user", "first")]}, config)
    app.invoke({"messages": [("user", "second")]}, config)

    # Walk from the head back to the root; it must reach every checkpoint.
    total = len(list(saver.list(config)))
    walked, cursor = 0, saver.get_tuple(config)
    while cursor is not None:
        walked += 1
        cursor = saver.get_tuple(cursor.parent_config) if cursor.parent_config else None

    assert walked == total, f"parent chain covers {walked} of {total} checkpoints"


def test_pending_writes_are_filed_under_the_checkpoint_id(saver, thread_id):
    """put_writes must key on checkpoint_id, never fall back to task_id.

    Writes filed under a task id are unreachable rather than absent — no read
    path ever looks there.
    """
    app = _build(DeltaState, saver)
    config = {"configurable": {"thread_id": thread_id}}
    app.invoke({"messages": [("user", "hi")]}, config)

    channels = {
        write[1]
        for tup in saver.list(config)
        for write in (tup.pending_writes or [])
    }
    assert "messages" in channels, "message writes were not retrievable by checkpoint"


def test_checkpoint_lookup_is_namespace_scoped(thread_id):
    """A checkpoint in one namespace must not answer a lookup in another.

    Bites as soon as subgraphs/subagents run, when a parent graph would
    otherwise resume from a child's checkpoint.
    """
    saver = KGraphMemoryCheckpointer()
    app = _build(DeltaState, saver)
    app.invoke({"messages": [("user", "hi")]}, {"configurable": {"thread_id": thread_id}})

    other_ns = {"configurable": {"thread_id": thread_id, "checkpoint_ns": "subagent"}}
    assert saver.get_tuple(other_ns) is None
    assert list(saver.list(other_ns)) == []
