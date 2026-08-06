"""
KGraph Redis Checkpointer - Redis-backed implementation of LangGraph checkpoint interface.

Uses only core Redis commands (SET, GET, DEL, SCAN, EXPIRE) — no Redis modules
(RedisJSON, RediSearch) required.  Compatible with AWS MemoryDB in production.

Local dev: redis://localhost:6381
Production: rediss://<memorydb-endpoint>  (TLS)
"""
import base64
import json
import logging
import os
from typing import Optional, Dict, Any, List, Iterator, Tuple, AsyncIterator
from contextlib import AbstractContextManager, AbstractAsyncContextManager

from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.types import Interrupt

from kgraphplanner.config.agent_config import AgentConfig, CheckpointConfig

try:
    import redis
    import redis.asyncio as aioredis
except ImportError:
    redis = None
    aioredis = None

logger = logging.getLogger(__name__)

# Fixed root prefix for all kgraphplanner checkpoint data
_ROOT = "kg"

# Marker stored in place of a channel blob when the channel has a new version
# but no value in ``channel_values``. This is how a ``DeltaChannel`` says
# "reconstruct me by walking the parent chain" rather than "I am empty".
_EMPTY_BLOB = "__kg_empty__"


class KGraphRedisCheckpointer(
    BaseCheckpointSaver[str],
    AbstractContextManager,
    AbstractAsyncContextManager,
):
    """
    Redis-backed checkpoint saver for KGraph Planner.

    Stores checkpoints, metadata, and pending writes as JSON blobs using
    basic SET/GET with TTL.  The caller's session ID is used directly as
    the LangGraph ``thread_id``.

    Parameters
    ----------
    redis_url : str, optional
        Redis connection URL.  Defaults to ``KGPLAN__CHECKPOINTING__REDIS_URL``
        or ``redis://localhost:6381``.
    ttl : int, optional
        Time-to-live in seconds for every key.  Defaults to
        ``KGPLAN__CHECKPOINTING__REDIS_TTL`` or 3600 (1 hour).
    serde : SerializerProtocol, optional
        Serializer for checkpoint/metadata objects (typically
        ``KGraphSerializer``).
    checkpoint_config : CheckpointConfig, optional
        Typed config object.  Falls back to ``AgentConfig.from_env()``.
    """

    def __init__(
        self,
        redis_url: str | None = None,
        ttl: int | None = None,
        serde: Optional[SerializerProtocol] = None,
        checkpoint_config: Optional[CheckpointConfig] = None,
    ):
        if redis is None:
            raise ImportError(
                "The 'redis' package is required for KGraphRedisCheckpointer. "
                "Install it with: pip install 'redis>=5.0.0'"
            )
        super().__init__(serde=serde)

        # Resolve config: explicit args > checkpoint_config > AgentConfig from env
        if checkpoint_config is None:
            checkpoint_config = AgentConfig.from_env().checkpointing

        self.redis_url = redis_url or checkpoint_config.redis_url
        self.ttl = ttl or checkpoint_config.redis_ttl
        self._sync_client: Optional[redis.Redis] = None
        self._async_client: Optional[aioredis.Redis] = None

        # Cluster mode: explicit config flag, or auto-detect from rediss:// scheme
        redis_cluster = checkpoint_config.redis_cluster
        if redis_cluster == "auto":
            self._cluster = self.redis_url.startswith("rediss://")
        else:
            self._cluster = redis_cluster in (True, "true", "True", "1")
        if self._cluster:
            logger.info(f"Redis cluster mode enabled (URL: {self.redis_url})")

        # Build key prefix: kg:{env}[:{app}]
        env = checkpoint_config.redis_env or "dev"
        app = checkpoint_config.redis_app or ""
        self._prefix = f"{_ROOT}:{env}:{app}" if app else f"{_ROOT}:{env}"
        logger.info(f"Redis key prefix: {self._prefix}")

    # ------------------------------------------------------------------
    # Lazy client accessors
    # ------------------------------------------------------------------

    def _get_sync_client(self) -> "redis.Redis":
        if self._sync_client is None:
            if self._cluster:
                self._sync_client = redis.RedisCluster.from_url(
                    self.redis_url, decode_responses=True
                )
            else:
                self._sync_client = redis.Redis.from_url(
                    self.redis_url, decode_responses=True
                )
            mode = "cluster" if self._cluster else "standalone"
            logger.info(f"Redis sync client ({mode}) connected to {self.redis_url}")
        return self._sync_client

    def _get_async_client(self) -> "aioredis.Redis":
        if self._async_client is None:
            if self._cluster:
                self._async_client = aioredis.RedisCluster.from_url(
                    self.redis_url, decode_responses=True
                )
            else:
                self._async_client = aioredis.Redis.from_url(
                    self.redis_url, decode_responses=True
                )
            mode = "cluster" if self._cluster else "standalone"
            logger.info(f"Redis async client ({mode}) connected to {self.redis_url}")
        return self._async_client

    # ------------------------------------------------------------------
    # Key generation
    # ------------------------------------------------------------------

    def _cp_key(self, thread_id: str, ns: str, checkpoint_id: str) -> str:
        return f"{self._prefix}:cp:{{{thread_id}}}:{ns}:{checkpoint_id}"

    def _meta_key(self, thread_id: str, ns: str, checkpoint_id: str) -> str:
        return f"{self._prefix}:meta:{{{thread_id}}}:{ns}:{checkpoint_id}"

    def _writes_key(self, thread_id: str, ns: str, checkpoint_id: str) -> str:
        return f"{self._prefix}:writes:{{{thread_id}}}:{ns}:{checkpoint_id}"

    def _latest_key(self, thread_id: str, ns: str) -> str:
        return f"{self._prefix}:latest:{{{thread_id}}}:{ns}"

    def _parent_key(self, thread_id: str, ns: str, checkpoint_id: str) -> str:
        return f"{self._prefix}:parent:{{{thread_id}}}:{ns}:{checkpoint_id}"

    def _blob_key(self, thread_id: str, ns: str, channel: str, version: Any) -> str:
        return f"{self._prefix}:blob:{{{thread_id}}}:{ns}:{channel}:{version}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_config(config: Dict[str, Any]) -> Tuple[str, str, Optional[str]]:
        """Return (thread_id, checkpoint_ns, checkpoint_id|None) from config."""
        c = config["configurable"]
        return (
            c["thread_id"],
            c.get("checkpoint_ns", ""),
            c.get("checkpoint_id"),
        )

    def _serialize(self, obj: Any) -> str:
        """Serialize a Python object to a Redis-storable string.

        Uses the typed serde API (``dumps_typed`` -> ``(type, payload)``). The
        untyped ``dumps``/``loads`` pair is not part of every serializer -- the
        default ``JsonPlusSerializer`` implements only the typed one -- and it
        round-trips LangChain messages into bare dicts, which breaks channel
        reconstruction.
        """
        if self.serde:
            type_, payload = self.serde.dumps_typed(obj)
            if isinstance(payload, (bytes, bytearray)):
                payload = base64.b64encode(payload).decode("ascii")
                return json.dumps([type_, payload, True])
            return json.dumps([type_, payload, False])
        return json.dumps(obj)

    def _deserialize(self, raw: str) -> Any:
        """Deserialize a Redis string produced by :meth:`_serialize`."""
        if self.serde:
            type_, payload, is_b64 = json.loads(raw)
            if is_b64:
                payload = base64.b64decode(payload.encode("ascii"))
            return self.serde.loads_typed((type_, payload))
        return json.loads(raw)

    def _serialize_write(self, channel: str, value: Any) -> str:
        """Serialize a single pending-write value.
        Interrupt objects use __slots__ so the generic serializer can't
        introspect them; convert to dicts explicitly."""
        if isinstance(value, Interrupt):
            value = {"__interrupt__": True, "value": value.value, "id": value.id}
        elif isinstance(value, (list, tuple)):
            value = [
                {"__interrupt__": True, "value": item.value, "id": item.id}
                if isinstance(item, Interrupt) else item
                for item in value
            ]
        return self._serialize(value)

    @staticmethod
    def _restore_interrupts(value: Any) -> Any:
        """Reconstruct Interrupt objects from tagged dicts after deserialization."""
        if isinstance(value, dict) and value.get("__interrupt__"):
            return Interrupt(value=value.get("value"), id=value.get("id", ""))
        if isinstance(value, list):
            return [
                Interrupt(value=item.get("value"), id=item.get("id", ""))
                if isinstance(item, dict) and item.get("__interrupt__")
                else item
                for item in value
            ]
        return value

    @staticmethod
    def _checkpoint_id(checkpoint: Any) -> str:
        """Extract the checkpoint id from a deserialized checkpoint."""
        if isinstance(checkpoint, dict):
            cid = checkpoint.get("id")
            if cid is None:
                raise ValueError("Checkpoint dict missing required 'id' key")
            return cid
        try:
            return checkpoint.id
        except AttributeError:
            raise ValueError(
                f"Checkpoint of type {type(checkpoint)} has no 'id' attribute"
            )

    @staticmethod
    def _channel_versions(checkpoint: Any) -> Dict[str, Any]:
        if isinstance(checkpoint, dict):
            return checkpoint.get("channel_versions", {}) or {}
        return getattr(checkpoint, "channel_versions", {}) or {}

    def _build_tuple(
        self,
        thread_id: str,
        ns: str,
        checkpoint_id: str,
        cp_raw: str,
        meta_raw: str,
        writes_raw: Optional[str],
        blob_raws: Dict[str, Optional[str]],
        parent_id: Optional[str],
    ) -> CheckpointTuple:
        checkpoint = self._deserialize(cp_raw)
        metadata = self._deserialize(meta_raw)

        # Reassemble channel_values from the per-(channel, version) blobs.
        channel_values: Dict[str, Any] = {}
        for channel, raw in blob_raws.items():
            if raw is None or raw == _EMPTY_BLOB:
                continue
            channel_values[channel] = self._deserialize(raw)
        if isinstance(checkpoint, dict):
            checkpoint = {**checkpoint, "channel_values": channel_values}

        pending_writes: List[Tuple[str, str, Any]] = []
        if writes_raw:
            for w in json.loads(writes_raw):
                task_id, channel, raw_val = w[0], w[1], w[2]
                value = self._deserialize(raw_val)
                # Reconstruct Interrupt objects after round-trip.
                # They may be stored as dicts (tagged with __interrupt__)
                # either bare or inside a list.
                value = self._restore_interrupts(value)
                pending_writes.append((task_id, channel, value))

        return CheckpointTuple(
            # Must identify the checkpoint actually loaded, not echo the caller's
            # config: the delta-channel ancestor walk resumes from here.
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": ns,
                    "checkpoint_id": checkpoint_id,
                }
            },
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": ns,
                        "checkpoint_id": parent_id,
                    }
                }
                if parent_id
                else None
            ),
            pending_writes=pending_writes,
        )

    def _merge_writes(
        self,
        existing_raw: Optional[str],
        writes: List[Tuple[str, Any]],
        task_id: str,
    ) -> str:
        """Append writes to the stored list, deduplicating retried tasks.

        Positive indices are append-once (a retried task must not duplicate
        them); the negative indices in ``WRITES_IDX_MAP`` (ERROR, INTERRUPT,
        RESUME, SCHEDULED) are upserts.
        """
        existing: List[List[Any]] = json.loads(existing_raw) if existing_raw else []
        # Index existing entries by the (task_id, idx) identity they were stored under.
        by_key = {(e[3], e[4]): i for i, e in enumerate(existing) if len(e) >= 5}
        for idx, (channel, value) in enumerate(writes):
            inner_idx = WRITES_IDX_MAP.get(channel, idx)
            key = (task_id, inner_idx)
            entry = [task_id, channel, self._serialize_write(channel, value), task_id, inner_idx]
            if key in by_key:
                if inner_idx >= 0:
                    continue  # append-once: already recorded
                existing[by_key[key]] = entry  # upsert
            else:
                by_key[key] = len(existing)
                existing.append(entry)
        return json.dumps(existing)

    # ------------------------------------------------------------------
    # Sync interface
    # ------------------------------------------------------------------

    @property
    def config_specs(self) -> List[Dict[str, Any]]:
        return [
            {
                "configurable": {
                    "thread_id": {"annotation": str, "default": None},
                    "checkpoint_ns": {"annotation": str, "default": ""},
                    "checkpoint_id": {"annotation": str, "default": None},
                }
            }
        ]

    def get_tuple(self, config: Dict[str, Any]) -> Optional[CheckpointTuple]:
        thread_id, ns, checkpoint_id = self._extract_config(config)
        r = self._get_sync_client()

        if not checkpoint_id:
            checkpoint_id = r.get(self._latest_key(thread_id, ns))
            if not checkpoint_id:
                logger.debug(f"No latest checkpoint for thread {thread_id}")
                return None

        cp_raw = r.get(self._cp_key(thread_id, ns, checkpoint_id))
        meta_raw = r.get(self._meta_key(thread_id, ns, checkpoint_id))
        if not cp_raw or not meta_raw:
            logger.debug(
                f"Checkpoint {checkpoint_id} not found for thread {thread_id}"
            )
            return None

        writes_raw = r.get(self._writes_key(thread_id, ns, checkpoint_id))
        parent_id = r.get(self._parent_key(thread_id, ns, checkpoint_id))
        blob_raws = self._load_blobs_sync(r, thread_id, ns, self._deserialize(cp_raw))

        logger.debug(f"Loaded checkpoint {checkpoint_id} for thread {thread_id}")
        return self._build_tuple(
            thread_id, ns, checkpoint_id, cp_raw, meta_raw, writes_raw, blob_raws, parent_id
        )

    def _load_blobs_sync(
        self, r: "redis.Redis", thread_id: str, ns: str, checkpoint: Any
    ) -> Dict[str, Optional[str]]:
        versions = self._channel_versions(checkpoint)
        if not versions:
            return {}
        channels = list(versions.keys())
        keys = [self._blob_key(thread_id, ns, ch, versions[ch]) for ch in channels]
        # MGET is not cross-slot safe in cluster mode; the {thread_id} hash tag
        # keeps every key for a thread in one slot, so this stays a single call.
        values = r.mget(keys)
        return dict(zip(channels, values))

    def list(
        self,
        config: Dict[str, Any],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        thread_id, ns, _ = self._extract_config(config)
        r = self._get_sync_client()

        pattern = self._cp_key(thread_id, ns, "*")
        keys: List[str] = []
        for key in r.scan_iter(match=pattern, count=100):
            keys.append(key)

        # Extract checkpoint_id from key, fetch and collect
        tuples = []
        for key in keys:
            # key format: kg:cp:{thread_id}:ns:checkpoint_id
            cp_id = key.rsplit(":", 1)[-1]
            cp_raw = r.get(key)
            meta_raw = r.get(self._meta_key(thread_id, ns, cp_id))
            if not cp_raw or not meta_raw:
                continue
            writes_raw = r.get(self._writes_key(thread_id, ns, cp_id))
            parent_id = r.get(self._parent_key(thread_id, ns, cp_id))
            blob_raws = self._load_blobs_sync(r, thread_id, ns, self._deserialize(cp_raw))
            tup = self._build_tuple(
                thread_id, ns, cp_id, cp_raw, meta_raw, writes_raw, blob_raws, parent_id
            )
            ts = (
                tup.checkpoint.get("ts", "")
                if isinstance(tup.checkpoint, dict)
                else getattr(tup.checkpoint, "ts", "")
            )
            tuples.append((ts, tup))

        # Sort by timestamp descending
        tuples.sort(key=lambda x: x[0], reverse=True)

        if before:
            tuples = [(ts, t) for ts, t in tuples if ts < before]

        if limit:
            tuples = tuples[:limit]

        for _, tup in tuples:
            yield tup

    def put(
        self,
        config: Dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> Dict[str, Any]:
        thread_id, ns, parent_id = self._extract_config(config)
        r = self._get_sync_client()

        if isinstance(checkpoint, dict) and "id" not in checkpoint:
            raise ValueError("Checkpoint dict missing required 'id' key")

        checkpoint_id = checkpoint["id"]
        c, values = self._split_channel_values(checkpoint)

        cp_blob = self._serialize(c)
        meta_blob = self._serialize(get_checkpoint_metadata(config, metadata))

        pipe = r.pipeline()
        self._queue_put(pipe, thread_id, ns, checkpoint_id, cp_blob, meta_blob,
                        parent_id, c, values, new_versions)
        pipe.execute()

        logger.debug(
            f"Saved checkpoint {checkpoint_id} for thread {thread_id} (TTL={self.ttl}s)"
        )
        # Contract: return a config identifying the checkpoint just written.
        # LangGraph threads this forward as the *next* put's parent, so
        # returning `config` unchanged severs the parent chain.
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    @staticmethod
    def _split_channel_values(checkpoint: Checkpoint) -> Tuple[Any, Dict[str, Any]]:
        """Separate the checkpoint skeleton from its channel values."""
        if isinstance(checkpoint, dict):
            c = dict(checkpoint)
            return c, c.pop("channel_values", {}) or {}
        return checkpoint, {}

    def _queue_put(self, pipe, thread_id, ns, checkpoint_id, cp_blob, meta_blob,
                   parent_id, checkpoint, values, new_versions) -> None:
        """Queue every write for one ``put`` onto a (sync or async) pipeline."""
        pipe.set(self._cp_key(thread_id, ns, checkpoint_id), cp_blob, ex=self.ttl)
        pipe.set(self._meta_key(thread_id, ns, checkpoint_id), meta_blob, ex=self.ttl)
        pipe.set(self._latest_key(thread_id, ns), checkpoint_id, ex=self.ttl)
        if parent_id:
            pipe.set(self._parent_key(thread_id, ns, checkpoint_id), parent_id, ex=self.ttl)

        # Channel values live in their own keys, addressed by (channel, version),
        # so an unchanged channel is shared across checkpoints rather than
        # recopied. A channel with a new version but no value is stored as
        # _EMPTY_BLOB: that is a DeltaChannel saying "reconstruct me from the
        # ancestor walk", which is not the same as "I am empty".
        for channel, version in (new_versions or {}).items():
            blob = self._serialize(values[channel]) if channel in values else _EMPTY_BLOB
            pipe.set(self._blob_key(thread_id, ns, channel, version), blob, ex=self.ttl)

        # Blobs are shared by version, so a channel untouched for longer than
        # the TTL would expire out from under the checkpoints still pointing at
        # it. Refresh every referenced blob on each write.
        for channel, version in self._channel_versions(checkpoint).items():
            if new_versions and channel in new_versions:
                continue
            pipe.expire(self._blob_key(thread_id, ns, channel, version), self.ttl)

    def put_writes(
        self,
        config: Dict[str, Any],
        writes: List[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id, ns, checkpoint_id = self._extract_config(config)
        # A task id is not a checkpoint id: filing writes under one puts them at
        # a key no read path will ever look up.
        checkpoint_id = config["configurable"]["checkpoint_id"]
        r = self._get_sync_client()

        key = self._writes_key(thread_id, ns, checkpoint_id)

        # Read-modify-write: append new writes to existing list.
        # Values are serialized individually since they may contain
        # non-JSON-serializable objects (e.g. HumanMessage).
        r.set(key, self._merge_writes(r.get(key), writes, task_id), ex=self.ttl)

    def delete_thread(self, thread_id: str) -> None:
        r = self._get_sync_client()
        pattern = f"{self._prefix}:*:{{{thread_id}}}:*"
        keys = list(r.scan_iter(match=pattern, count=100))
        if keys:
            r.delete(*keys)
            logger.debug(f"Deleted {len(keys)} keys for thread {thread_id}")

    # ------------------------------------------------------------------
    # Async interface
    # ------------------------------------------------------------------

    async def aget_tuple(
        self, config: Dict[str, Any]
    ) -> Optional[CheckpointTuple]:
        thread_id, ns, checkpoint_id = self._extract_config(config)
        r = self._get_async_client()

        if not checkpoint_id:
            checkpoint_id = await r.get(self._latest_key(thread_id, ns))
            if not checkpoint_id:
                logger.debug(f"No latest checkpoint for thread {thread_id}")
                return None

        cp_raw = await r.get(self._cp_key(thread_id, ns, checkpoint_id))
        meta_raw = await r.get(self._meta_key(thread_id, ns, checkpoint_id))
        if not cp_raw or not meta_raw:
            logger.debug(
                f"Checkpoint {checkpoint_id} not found for thread {thread_id}"
            )
            return None

        writes_raw = await r.get(self._writes_key(thread_id, ns, checkpoint_id))
        parent_id = await r.get(self._parent_key(thread_id, ns, checkpoint_id))
        blob_raws = await self._load_blobs_async(r, thread_id, ns, self._deserialize(cp_raw))

        logger.debug(f"Loaded checkpoint {checkpoint_id} for thread {thread_id}")
        return self._build_tuple(
            thread_id, ns, checkpoint_id, cp_raw, meta_raw, writes_raw, blob_raws, parent_id
        )

    async def _load_blobs_async(
        self, r: "aioredis.Redis", thread_id: str, ns: str, checkpoint: Any
    ) -> Dict[str, Optional[str]]:
        versions = self._channel_versions(checkpoint)
        if not versions:
            return {}
        channels = list(versions.keys())
        keys = [self._blob_key(thread_id, ns, ch, versions[ch]) for ch in channels]
        values = await r.mget(keys)
        return dict(zip(channels, values))

    async def alist(
        self,
        config: Dict[str, Any],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[CheckpointTuple]:
        thread_id, ns, _ = self._extract_config(config)
        r = self._get_async_client()

        pattern = self._cp_key(thread_id, ns, "*")
        keys: List[str] = []
        async for key in r.scan_iter(match=pattern, count=100):
            keys.append(key)

        tuples = []
        for key in keys:
            cp_id = key.rsplit(":", 1)[-1]
            cp_raw = await r.get(key)
            meta_raw = await r.get(self._meta_key(thread_id, ns, cp_id))
            if not cp_raw or not meta_raw:
                continue
            writes_raw = await r.get(self._writes_key(thread_id, ns, cp_id))
            parent_id = await r.get(self._parent_key(thread_id, ns, cp_id))
            blob_raws = await self._load_blobs_async(
                r, thread_id, ns, self._deserialize(cp_raw)
            )
            tup = self._build_tuple(
                thread_id, ns, cp_id, cp_raw, meta_raw, writes_raw, blob_raws, parent_id
            )
            ts = (
                tup.checkpoint.get("ts", "")
                if isinstance(tup.checkpoint, dict)
                else getattr(tup.checkpoint, "ts", "")
            )
            tuples.append((ts, tup))

        tuples.sort(key=lambda x: x[0], reverse=True)

        if before:
            tuples = [(ts, t) for ts, t in tuples if ts < before]
        if limit:
            tuples = tuples[:limit]

        return [t for _, t in tuples]

    async def aput(
        self,
        config: Dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> Dict[str, Any]:
        thread_id, ns, parent_id = self._extract_config(config)
        r = self._get_async_client()

        if isinstance(checkpoint, dict) and "id" not in checkpoint:
            raise ValueError("Checkpoint dict missing required 'id' key")

        checkpoint_id = checkpoint["id"]
        c, values = self._split_channel_values(checkpoint)

        cp_blob = self._serialize(c)
        meta_blob = self._serialize(get_checkpoint_metadata(config, metadata))

        pipe = r.pipeline()
        self._queue_put(pipe, thread_id, ns, checkpoint_id, cp_blob, meta_blob,
                        parent_id, c, values, new_versions)
        await pipe.execute()

        logger.debug(
            f"Saved checkpoint {checkpoint_id} for thread {thread_id} (TTL={self.ttl}s)"
        )
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    async def aput_writes(
        self,
        config: Dict[str, Any],
        writes: List[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id, ns, _ = self._extract_config(config)
        checkpoint_id = config["configurable"]["checkpoint_id"]
        r = self._get_async_client()

        key = self._writes_key(thread_id, ns, checkpoint_id)

        merged = self._merge_writes(await r.get(key), writes, task_id)
        await r.set(key, merged, ex=self.ttl)

    async def adelete_thread(self, thread_id: str) -> None:
        r = self._get_async_client()
        pattern = f"{self._prefix}:*:{{{thread_id}}}:*"
        keys: List[str] = []
        async for key in r.scan_iter(match=pattern, count=100):
            keys.append(key)
        if keys:
            await r.delete(*keys)
            logger.debug(f"Deleted {len(keys)} keys for thread {thread_id}")

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._sync_client:
            self._sync_client.close()
            self._sync_client = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._async_client:
            await self._async_client.aclose()
            self._async_client = None
