"""
KGraph Redis Checkpointer - Redis-backed implementation of LangGraph checkpoint interface.

Uses only core Redis commands (SET, GET, DEL, SCAN, EXPIRE) — no Redis modules
(RedisJSON, RediSearch) required.  Compatible with AWS MemoryDB in production.

Local dev: redis://localhost:6381
Production: rediss://<memorydb-endpoint>  (TLS)
"""
import json
import logging
import os
from typing import Optional, Dict, Any, List, Iterator, Tuple, AsyncIterator
from contextlib import AbstractContextManager, AbstractAsyncContextManager

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
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

    def _deserialize(self, raw: str) -> Any:
        """Deserialize a Redis string back to a Python object.
        Uses serde.loads() which accepts the bytes produced by serde.dumps()."""
        if self.serde:
            return self.serde.loads(raw.encode("utf-8"))
        return json.loads(raw)

    def _serialize(self, obj: Any) -> str:
        """Serialize a Python object to a Redis-storable string.
        Uses serde.dumps() which returns base64-encoded UTF-8 bytes."""
        if self.serde:
            return self.serde.dumps(obj).decode("utf-8")
        return json.dumps(obj)

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

    def _build_tuple(
        self,
        config: Dict[str, Any],
        cp_raw: str,
        meta_raw: str,
        writes_raw: Optional[str],
        parent_config: Optional[Dict[str, Any]] = None,
    ) -> CheckpointTuple:
        checkpoint = self._deserialize(cp_raw)
        metadata = self._deserialize(meta_raw)
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
            config=config,
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=pending_writes,
        )

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

        logger.debug(f"Loaded checkpoint {checkpoint_id} for thread {thread_id}")
        return self._build_tuple(config, cp_raw, meta_raw, writes_raw)

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
            tup = self._build_tuple(config, cp_raw, meta_raw, writes_raw)
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
        new_versions: Dict[str, Any],
    ) -> Dict[str, Any]:
        thread_id, ns, _ = self._extract_config(config)
        r = self._get_sync_client()

        if isinstance(checkpoint, dict) and "id" not in checkpoint:
            raise ValueError("Checkpoint dict missing required 'id' key")

        if isinstance(metadata, dict):
            metadata.setdefault("source", "update")
            metadata.setdefault("step", -1)

        checkpoint_id = checkpoint["id"]

        cp_blob = self._serialize(checkpoint)
        meta_blob = self._serialize(metadata)

        pipe = r.pipeline()
        pipe.set(self._cp_key(thread_id, ns, checkpoint_id), cp_blob, ex=self.ttl)
        pipe.set(self._meta_key(thread_id, ns, checkpoint_id), meta_blob, ex=self.ttl)
        pipe.set(self._latest_key(thread_id, ns), checkpoint_id, ex=self.ttl)
        pipe.execute()

        logger.debug(
            f"Saved checkpoint {checkpoint_id} for thread {thread_id} (TTL={self.ttl}s)"
        )
        return config

    def put_writes(
        self,
        config: Dict[str, Any],
        writes: List[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        thread_id, ns, checkpoint_id = self._extract_config(config)
        checkpoint_id = checkpoint_id or task_id
        r = self._get_sync_client()

        key = self._writes_key(thread_id, ns, checkpoint_id)

        # Read-modify-write: append new writes to existing list.
        # Values are serialized individually since they may contain
        # non-JSON-serializable objects (e.g. HumanMessage).
        existing_raw = r.get(key)
        existing: List[List[Any]] = json.loads(existing_raw) if existing_raw else []
        existing.extend(
            [task_id, k, self._serialize_write(k, v)] for k, v in writes
        )

        r.set(key, json.dumps(existing), ex=self.ttl)

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

        logger.debug(f"Loaded checkpoint {checkpoint_id} for thread {thread_id}")
        return self._build_tuple(config, cp_raw, meta_raw, writes_raw)

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
            tup = self._build_tuple(config, cp_raw, meta_raw, writes_raw)
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
        new_versions: Dict[str, Any],
    ) -> Dict[str, Any]:
        thread_id, ns, _ = self._extract_config(config)
        r = self._get_async_client()

        if isinstance(checkpoint, dict) and "id" not in checkpoint:
            raise ValueError("Checkpoint dict missing required 'id' key")

        if isinstance(metadata, dict):
            metadata.setdefault("source", "update")
            metadata.setdefault("step", -1)

        checkpoint_id = checkpoint["id"]

        cp_blob = self._serialize(checkpoint)
        meta_blob = self._serialize(metadata)

        pipe = r.pipeline()
        pipe.set(self._cp_key(thread_id, ns, checkpoint_id), cp_blob, ex=self.ttl)
        pipe.set(self._meta_key(thread_id, ns, checkpoint_id), meta_blob, ex=self.ttl)
        pipe.set(self._latest_key(thread_id, ns), checkpoint_id, ex=self.ttl)
        await pipe.execute()

        logger.debug(
            f"Saved checkpoint {checkpoint_id} for thread {thread_id} (TTL={self.ttl}s)"
        )
        return config

    async def aput_writes(
        self,
        config: Dict[str, Any],
        writes: List[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        thread_id, ns, checkpoint_id = self._extract_config(config)
        checkpoint_id = checkpoint_id or task_id
        r = self._get_async_client()

        key = self._writes_key(thread_id, ns, checkpoint_id)

        existing_raw = await r.get(key)
        existing: List[List[Any]] = json.loads(existing_raw) if existing_raw else []
        existing.extend(
            [task_id, k, self._serialize_write(k, v)] for k, v in writes
        )

        await r.set(key, json.dumps(existing), ex=self.ttl)

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
