"""
KGraph Memory Checkpointer - In-memory implementation of LangGraph checkpoint interface
"""
import logging
import asyncio
from typing import Optional, Dict, Any, List, Iterator, Sequence, Tuple
from collections import defaultdict
from contextlib import AbstractContextManager, AbstractAsyncContextManager
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.serde.base import SerializerProtocol

# Set up logger
logger = logging.getLogger(__name__)


class KGraphMemoryCheckpointer(BaseCheckpointSaver[str], AbstractContextManager, AbstractAsyncContextManager):
    """
    In-memory checkpoint saver for KGraph Planner.

    This implementation stores checkpoints in memory using dictionaries.
    Only use for debugging, testing, or development purposes.

    Storage layout mirrors ``langgraph.checkpoint.memory.InMemorySaver``:

    - ``storage[thread_id][checkpoint_ns][checkpoint_id]`` -> (checkpoint, metadata, parent_id)
    - ``blobs[(thread_id, checkpoint_ns, channel, version)]`` -> serialized channel value
    - ``writes[(thread_id, checkpoint_ns, checkpoint_id)][(task_id, idx)]`` -> pending write

    Channel values are stored per (channel, version) rather than inline in the
    checkpoint, and reassembled on read from ``channel_versions``. Checkpoints
    also record their parent id: LangGraph's ``DeltaChannel`` support (used by
    ``create_deep_agent`` for ``messages`` and ``files``) omits those channels
    from ``channel_values`` between snapshots and reconstructs them by walking
    the parent chain via ``BaseCheckpointSaver.get_delta_channel_history``.
    Without the parent link that walk terminates immediately and the thread
    silently comes back with no messages.
    """

    def __init__(self, serde: Optional[SerializerProtocol] = None):
        """Initialize the in-memory checkpointer."""
        super().__init__(serde=serde)
        # thread_id -> checkpoint_ns -> checkpoint_id -> (checkpoint, metadata, parent_id)
        self.storage: Dict[str, Dict[str, Dict[str, Tuple[Any, Any, Optional[str]]]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        # (thread_id, checkpoint_ns, checkpoint_id) -> (task_id, idx) -> (task_id, channel, value, task_path)
        self.writes: Dict[Tuple[str, str, str], Dict[Tuple[str, int], Tuple[str, str, Any, str]]] = defaultdict(dict)
        # (thread_id, checkpoint_ns, channel, version) -> serialized value
        self.blobs: Dict[Tuple[str, str, str, Any], Tuple[str, bytes]] = {}

    @property
    def config_specs(self) -> List[Dict[str, Any]]:
        """Define the configuration options for the checkpoint saver."""
        return [
            {
                "configurable": {
                    "thread_id": {"annotation": str, "default": None},
                    "checkpoint_ns": {"annotation": str, "default": ""},
                    "checkpoint_id": {"annotation": str, "default": None},
                }
            }
        ]

    def _load_blobs(self, thread_id: str, checkpoint_ns: str, versions: ChannelVersions) -> Dict[str, Any]:
        """Reassemble channel_values from the per-(channel, version) blob store."""
        result: Dict[str, Any] = {}
        for channel, version in versions.items():
            blob = self.blobs.get((thread_id, checkpoint_ns, channel, version))
            if blob is None or blob[0] == "empty":
                continue
            result[channel] = self.serde.loads_typed(blob)
        return result

    def _parent_config(self, thread_id: str, checkpoint_ns: str, parent_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if not parent_id:
            return None
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": parent_id,
            }
        }

    def _build_tuple(self, thread_id: str, checkpoint_ns: str, checkpoint_id: str) -> CheckpointTuple:
        checkpoint_raw, metadata_raw, parent_id = self.storage[thread_id][checkpoint_ns][checkpoint_id]
        checkpoint: Checkpoint = self.serde.loads_typed(checkpoint_raw)
        writes = self.writes.get((thread_id, checkpoint_ns, checkpoint_id), {}).values()
        return CheckpointTuple(
            # Must identify the checkpoint that was actually loaded, not echo the
            # caller's config: the delta-channel ancestor walk resumes from here.
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            },
            checkpoint={
                **checkpoint,
                "channel_values": self._load_blobs(thread_id, checkpoint_ns, checkpoint["channel_versions"]),
            },
            metadata=self.serde.loads_typed(metadata_raw),
            parent_config=self._parent_config(thread_id, checkpoint_ns, parent_id),
            pending_writes=[(tid, ch, self.serde.loads_typed(v)) for tid, ch, v, _ in writes],
        )

    def get_tuple(self, config: Dict[str, Any]) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the in-memory storage."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        logger.debug(f"get_tuple called for thread_id={thread_id}, ns={checkpoint_ns!r}")

        # Namespace-scoped: a checkpoint in one namespace must never be
        # returned for a lookup in another (subgraphs / subagents).
        ns_storage = self.storage.get(thread_id, {}).get(checkpoint_ns, {})
        if not ns_storage:
            logger.debug(f"No checkpoints for thread {thread_id} in ns {checkpoint_ns!r}")
            return None

        checkpoint_id = get_checkpoint_id(config)
        if checkpoint_id:
            if checkpoint_id not in ns_storage:
                return None
        else:
            # Checkpoint ids are UUID6 (time-ordered), so max() is the latest.
            checkpoint_id = max(ns_storage.keys())

        return self._build_tuple(thread_id, checkpoint_ns, checkpoint_id)

    def list(self, config: Dict[str, Any], *, filter: Optional[Dict[str, Any]] = None, before: Optional[Any] = None, limit: Optional[int] = None) -> Iterator[CheckpointTuple]:
        """List checkpoints from the in-memory storage, most recent first."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        ns_storage = self.storage.get(thread_id, {}).get(checkpoint_ns, {})
        if not ns_storage:
            return

        before_id = get_checkpoint_id(before) if before else None

        for checkpoint_id in sorted(ns_storage.keys(), reverse=True):
            if before_id and checkpoint_id >= before_id:
                continue

            tup = self._build_tuple(thread_id, checkpoint_ns, checkpoint_id)

            if filter and not all(tup.metadata.get(k) == v for k, v in filter.items()):
                continue

            if limit is not None:
                if limit <= 0:
                    break
                limit -= 1

            yield tup

    def put(self, config: Dict[str, Any], checkpoint: Checkpoint, metadata: CheckpointMetadata, new_versions: ChannelVersions) -> Dict[str, Any]:
        """Save a checkpoint to the in-memory storage."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        if isinstance(checkpoint, dict) and "id" not in checkpoint:
            raise ValueError("Checkpoint dictionary does not contain required 'id' key")

        if not self.serde:
            raise ValueError("Serializer is required for checkpoint storage")

        checkpoint_id = checkpoint["id"]

        # Channel values live in the blob store, keyed by (channel, version), so
        # an unchanged channel is shared across checkpoints rather than recopied.
        # A channel present in new_versions but absent from channel_values is
        # recorded as "empty" — that is how a DeltaChannel signals "reconstruct
        # me from the ancestor walk".
        c = dict(checkpoint)
        values: Dict[str, Any] = c.pop("channel_values", {})
        for channel, version in new_versions.items():
            self.blobs[(thread_id, checkpoint_ns, channel, version)] = (
                self.serde.dumps_typed(values[channel]) if channel in values else ("empty", b"")
            )

        self.storage[thread_id][checkpoint_ns][checkpoint_id] = (
            self.serde.dumps_typed(c),
            self.serde.dumps_typed(get_checkpoint_metadata(config, metadata)),
            config["configurable"].get("checkpoint_id"),  # parent
        )

        # Contract: return a config identifying the checkpoint just written.
        # LangGraph threads this forward as the *next* put's parent, so
        # returning `config` unchanged severs the parent chain.
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(self, config: Dict[str, Any], writes: Sequence[Tuple[str, Any]], task_id: str, task_path: str = "") -> None:
        """Store intermediate writes linked to a checkpoint."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        # A task id is not a checkpoint id: filing writes under one puts them at
        # a key no read path will ever look up.
        checkpoint_id = config["configurable"]["checkpoint_id"]

        outer_key = (thread_id, checkpoint_ns, checkpoint_id)
        existing = self.writes.get(outer_key)
        for idx, (channel, value) in enumerate(writes):
            inner_key = (task_id, WRITES_IDX_MAP.get(channel, idx))
            # Negative indices are upserts (ERROR/INTERRUPT/RESUME); positive
            # ones are append-once, so a retried task must not duplicate them.
            if inner_key[1] >= 0 and existing and inner_key in existing:
                continue
            self.writes[outer_key][inner_key] = (
                task_id,
                channel,
                self.serde.dumps_typed(value),
                task_path,
            )

    def delete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints, writes and blobs associated with a thread ID."""
        if thread_id in self.storage:
            del self.storage[thread_id]
        for key in list(self.writes.keys()):
            if key[0] == thread_id:
                del self.writes[key]
        for key in list(self.blobs.keys()):
            if key[0] == thread_id:
                del self.blobs[key]

    # Async versions — offload to thread to avoid blocking the event loop.
    # For in-memory dicts this is fast, but correctness matters when the
    # checkpointer is used inside an async LangGraph agent.
    async def aget_tuple(self, config: Dict[str, Any]) -> Optional[CheckpointTuple]:
        """Asynchronous version of get_tuple."""
        return await asyncio.to_thread(self.get_tuple, config)

    async def alist(self, config: Dict[str, Any], *, filter: Optional[Dict[str, Any]] = None, before: Optional[Any] = None, limit: Optional[int] = None) -> List[CheckpointTuple]:
        """Asynchronous version of list."""
        return await asyncio.to_thread(lambda: list(self.list(config, filter=filter, before=before, limit=limit)))

    async def aput(self, config: Dict[str, Any], checkpoint: Checkpoint, metadata: CheckpointMetadata, new_versions: ChannelVersions) -> Dict[str, Any]:
        """Asynchronous version of put."""
        return await asyncio.to_thread(self.put, config, checkpoint, metadata, new_versions)

    async def aput_writes(self, config: Dict[str, Any], writes: Sequence[Tuple[str, Any]], task_id: str, task_path: str = "") -> None:
        """Asynchronous version of put_writes."""
        return await asyncio.to_thread(self.put_writes, config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        """Asynchronous version of delete_thread."""
        await asyncio.to_thread(self.delete_thread, thread_id)

    # Context manager methods
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
