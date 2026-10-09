"""
KGraph Custom Serializer - Custom serialization for checkpoints
"""
import json
import logging
from typing import Any, Optional, Tuple

from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from kgraphplanner.checkpointer.kgraph_serialized import KGraphSerialized

# Set up logger
logger = logging.getLogger(__name__)

# dumps_typed() type tags look like "kgraph+<inner type>", e.g. "kgraph+msgpack".
TYPE_PREFIX = "kgraph+"


class KGraphSerializer(SerializerProtocol):
    """
    Checkpoint serializer that wraps LangGraph's serializer in KGraph's
    base64 ``KGraphSerialized`` encoding.

    Implements LangGraph's typed protocol: ``dumps_typed(obj)`` returns
    ``(type, bytes)`` and ``loads_typed((type, bytes))`` reverses it, which is
    what ``KGraphMemoryCheckpointer`` and ``KGraphRedisCheckpointer`` call. The
    value itself is encoded by ``JsonPlusSerializer`` (or ``inner``), so
    LangChain messages, tool calls and other state round-trip as the same
    types instead of degrading to plain JSON.

    ``loads_typed`` also reads a bare ``KGraphSerialized`` object, the format
    earlier versions wrote.
    """

    def __init__(self, inner: Optional[SerializerProtocol] = None):
        self._inner = inner or JsonPlusSerializer()
        logger.info("KGraphSerializer initialized")

    # --- Typed protocol (used by the checkpointers) -------------------------

    def dumps_typed(self, obj: Any) -> Tuple[str, bytes]:
        """Serialize ``obj`` to ``("kgraph+<inner type>", base64 bytes)``."""
        inner_type, data = self._inner.dumps_typed(obj)
        if isinstance(data, str):
            data = data.encode("utf-8")
        container = KGraphSerialized(data=data, original_type=type(obj).__name__)
        return f"{TYPE_PREFIX}{inner_type}", container.encoded_data.encode("ascii")

    def loads_typed(self, data: Any) -> Any:
        """Deserialize what ``dumps_typed`` produced."""
        if isinstance(data, KGraphSerialized):
            # Written by earlier versions: base64-encoded JSON, no type tag.
            return json.loads(data.get_data().decode("utf-8"))
        type_, payload = data
        if type_.startswith(TYPE_PREFIX):
            raw = KGraphSerialized.from_bytes(bytes(payload)).get_data()
            return self._inner.loads_typed((type_[len(TYPE_PREFIX):], raw))
        # Not ours (e.g. written by a different serde): let the inner one try.
        return self._inner.loads_typed((type_, payload))

    # --- Untyped JSON helpers (kept for existing callers) -------------------

    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format."""
        if hasattr(obj, 'model_dump'):
            # Pydantic models
            return obj.model_dump()
        elif hasattr(obj, 'to_dict'):
            # Objects with to_dict method
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):
            # Regular objects - recursively serialize their attributes
            result = {}
            for key, value in obj.__dict__.items():
                result[key] = self._make_json_serializable(value)
            return result
        elif isinstance(obj, dict):
            # Dictionaries - recursively serialize values
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            # Lists/tuples - recursively serialize elements
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            # Already JSON-serializable
            return obj
        else:
            # Convert to string representation as fallback
            return str(obj)

    def dumps(self, obj: Any) -> bytes:
        """Serialize to base64-encoded JSON bytes. Lossy for non-JSON types:
        checkpointers use ``dumps_typed``."""
        json_bytes = json.dumps(self._make_json_serializable(obj), ensure_ascii=False,
                                separators=(',', ':')).encode('utf-8')
        return KGraphSerialized(data=json_bytes, original_type=type(obj).__name__).encoded_data.encode('utf-8')

    def loads(self, data: bytes) -> Any:
        """Load what ``dumps`` produced."""
        if not isinstance(data, bytes):
            raise TypeError(f"Expected bytes, got {type(data).__name__}")
        return json.loads(KGraphSerialized.from_bytes(data).get_data().decode('utf-8'))
