"""
KGraph Memory Checkpointer - In-memory implementation of LangGraph checkpoint interface
"""
import logging
import asyncio
from typing import Optional, Dict, Any, List, Iterator, Tuple
from collections import defaultdict
from contextlib import AbstractContextManager, AbstractAsyncContextManager
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointMetadata, CheckpointTuple
from langgraph.checkpoint.serde.base import SerializerProtocol

# Set up logger
logger = logging.getLogger(__name__)


class KGraphMemoryCheckpointer(BaseCheckpointSaver[str], AbstractContextManager, AbstractAsyncContextManager):
    """
    In-memory checkpoint saver for KGraph Planner.
    
    This implementation stores checkpoints in memory using dictionaries.
    Only use for debugging, testing, or development purposes.
    """
    
    def __init__(self, serde: Optional[SerializerProtocol] = None):
        """Initialize the in-memory checkpointer."""
        super().__init__(serde=serde)
        self.storage: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.writes: Dict[str, Dict[str, List[Tuple[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    
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
    
    def get_tuple(self, config: Dict[str, Any]) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the in-memory storage."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"].get("checkpoint_id")
        
        logger.debug(f"get_tuple called for thread_id={thread_id}, checkpoint_id={checkpoint_id}")
        logger.debug(f"Available threads: {list(self.storage.keys())}")
        
        if thread_id not in self.storage:
            logger.debug(f"Thread {thread_id} not found in storage")
            return None
            
        thread_data = self.storage[thread_id]
        
        if checkpoint_id:
            # Get specific checkpoint
            key = f"{checkpoint_ns}:{checkpoint_id}"
            if key not in thread_data:
                return None
            checkpoint_data = thread_data[key]
        else:
            # Get latest checkpoint
            if not thread_data:
                return None
            # Find the most recent checkpoint
            def get_checkpoint_ts(key):
                checkpoint_data = thread_data[key]
                # Deserialize checkpoint to get timestamp
                if self.serde:
                    try:
                        checkpoint = self.serde.loads_typed(checkpoint_data["checkpoint"])
                        return checkpoint.get("ts", "") if isinstance(checkpoint, dict) else getattr(checkpoint, 'ts', "")
                    except Exception:
                        return ""
                else:
                    # Fallback for non-serialized data
                    checkpoint = checkpoint_data.get("checkpoint", {})
                    return checkpoint.get("ts", "") if isinstance(checkpoint, dict) else getattr(checkpoint, 'ts', "")
            
            latest_key = max(thread_data.keys(), key=get_checkpoint_ts)
            checkpoint_data = thread_data[latest_key]
        
        # Deserialize checkpoint and metadata if serde is available
        if self.serde:
            checkpoint = self.serde.loads_typed(checkpoint_data["checkpoint"])
            metadata = self.serde.loads_typed(checkpoint_data["metadata"])
        else:
            # Fallback to creating objects from stored data
            checkpoint = Checkpoint(**checkpoint_data["checkpoint"])
            metadata = CheckpointMetadata(**checkpoint_data["metadata"])
        
        parent_config = checkpoint_data.get("parent_config")
        
        # Get pending writes - checkpoint should have 'id' key (it's a TypedDict/dict)
        if isinstance(checkpoint, dict):
            checkpoint_id = checkpoint.get('id')
            if checkpoint_id is None:
                raise ValueError(f"Deserialized checkpoint dictionary does not contain required 'id' key")
        else:
            try:
                checkpoint_id = checkpoint.id
            except AttributeError:
                raise ValueError(f"Deserialized checkpoint object of type {type(checkpoint)} does not have required 'id' attribute")
        
        writes_key = f"{checkpoint_ns}:{checkpoint_id}"
        pending_writes = self.writes[thread_id].get(writes_key, [])
        
        return CheckpointTuple(
            config=config,
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=pending_writes
        )
    
    def list(self, config: Dict[str, Any], *, filter: Optional[Dict[str, Any]] = None, before: Optional[str] = None, limit: Optional[int] = None) -> Iterator[CheckpointTuple]:
        """List checkpoints from the in-memory storage."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        
        if thread_id not in self.storage:
            return
            
        thread_data = self.storage[thread_id]
        
        # Filter by namespace if specified
        items = []
        for key, data in thread_data.items():
            if key.startswith(f"{checkpoint_ns}:"):
                items.append((key, data))
        
        # Sort by timestamp (most recent first)
        items.sort(key=lambda x: x[1]["checkpoint"]["ts"], reverse=True)
        
        # Apply before filter
        if before:
            items = [item for item in items if item[1]["checkpoint"]["ts"] < before]
        
        # Apply limit
        if limit:
            items = items[:limit]
        
        # Yield checkpoint tuples
        for key, data in items:
            # Deserialize checkpoint and metadata if serde is available
            if self.serde:
                checkpoint = self.serde.loads_typed(data["checkpoint"])
                metadata = self.serde.loads_typed(data["metadata"])
            else:
                checkpoint = Checkpoint(**data["checkpoint"])
                metadata = CheckpointMetadata(**data["metadata"])
            
            parent_config = data.get("parent_config")
            
            # Get pending writes - checkpoint should always have .id attribute
            try:
                checkpoint_id = checkpoint.id
            except AttributeError:
                raise ValueError(f"Deserialized checkpoint object of type {type(checkpoint)} does not have required 'id' attribute")
            
            writes_key = f"{checkpoint_ns}:{checkpoint_id}"
            pending_writes = self.writes[thread_id].get(writes_key, [])
            
            yield CheckpointTuple(
                config=config,
                checkpoint=checkpoint,
                metadata=metadata,
                parent_config=parent_config,
                pending_writes=pending_writes
            )
    
    def put(self, config: Dict[str, Any], checkpoint: Checkpoint, metadata: CheckpointMetadata, new_versions: Dict[str, Any]) -> Dict[str, Any]:
        """Save a checkpoint to the in-memory storage."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        
        # Checkpoint is a TypedDict, so it's already a dict - just ensure it has required fields
        if isinstance(checkpoint, dict):
            # Ensure checkpoint has required 'id' field
            if 'id' not in checkpoint:
                raise ValueError(f"Checkpoint dictionary does not contain required 'id' key")
        
        if isinstance(metadata, dict):
            # Ensure metadata has required structure for CheckpointMetadata TypedDict
            if 'source' not in metadata:
                metadata['source'] = 'update'
            if 'step' not in metadata:
                metadata['step'] = -1
        
        # Get checkpoint ID from dictionary
        checkpoint_id = checkpoint['id']
        
        key = f"{checkpoint_ns}:{checkpoint_id}"
        
        # Serialize checkpoint and metadata using our serializer
        if not self.serde:
            raise ValueError("Serializer is required for checkpoint storage")
        
        checkpoint_data = self.serde.dumps_typed(checkpoint)
        metadata_data = self.serde.dumps_typed(metadata)
        
        self.storage[thread_id][key] = {
            "checkpoint": checkpoint_data,
            "metadata": metadata_data,
            "new_versions": new_versions
        }
        
        return config
    
    def put_writes(self, config: Dict[str, Any], writes: List[Tuple[str, Any]], task_id: str) -> None:
        """Store intermediate writes linked to a checkpoint.
        
        LangGraph expects pending_writes as 3-tuples (task_id, channel, value).
        The `writes` arg arrives as 2-tuples (channel, value); we prepend task_id.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"].get("checkpoint_id", task_id)
        
        key = f"{checkpoint_ns}:{checkpoint_id}"
        self.writes[thread_id][key].extend(
            (task_id, k, v) for k, v in writes
        )
    
    def delete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints and writes associated with a thread ID."""
        if thread_id in self.storage:
            del self.storage[thread_id]
        if thread_id in self.writes:
            del self.writes[thread_id]
    
    # Async versions — offload to thread to avoid blocking the event loop.
    # For in-memory dicts this is fast, but correctness matters when the
    # checkpointer is used inside an async LangGraph agent.
    async def aget_tuple(self, config: Dict[str, Any]) -> Optional[CheckpointTuple]:
        """Asynchronous version of get_tuple."""
        return await asyncio.to_thread(self.get_tuple, config)
    
    async def alist(self, config: Dict[str, Any], *, filter: Optional[Dict[str, Any]] = None, before: Optional[str] = None, limit: Optional[int] = None) -> List[CheckpointTuple]:
        """Asynchronous version of list."""
        return await asyncio.to_thread(lambda: list(self.list(config, filter=filter, before=before, limit=limit)))
    
    async def aput(self, config: Dict[str, Any], checkpoint: Checkpoint, metadata: CheckpointMetadata, new_versions: Dict[str, Any]) -> Dict[str, Any]:
        """Asynchronous version of put."""
        return await asyncio.to_thread(self.put, config, checkpoint, metadata, new_versions)
    
    async def aput_writes(self, config: Dict[str, Any], writes: List[Tuple[str, Any]], task_id: str) -> None:
        """Asynchronous version of put_writes."""
        return await asyncio.to_thread(self.put_writes, config, writes, task_id)
    
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