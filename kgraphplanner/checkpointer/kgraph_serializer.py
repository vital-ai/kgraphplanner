"""
KGraph Custom Serializer - Custom serialization for checkpoints
"""
import json
import logging
from typing import Any
from langgraph.checkpoint.serde.base import SerializerProtocol
from kgraphplanner.checkpointer.kgraph_serialized import KGraphSerialized

# Set up logger
logger = logging.getLogger(__name__)

class KGraphSerializer(SerializerProtocol):
    """
    Serializer that orchestrates KGraphSerialized containers.
    Handles smart data preparation before serialization.
    """
    
    def __init__(self):
        logger.info("KGraphSerializer initialized")
    
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
        """Serialize object and return as bytes."""
        # Do the smart conversion work first
        json_serializable_data = self._make_json_serializable(obj)
        # Convert to bytes
        json_bytes = json.dumps(json_serializable_data, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        # Then let KGraphSerialized handle pure encoding
        serialized = KGraphSerialized(data=json_bytes, original_type=type(obj).__name__)
        return serialized.encoded_data.encode('utf-8')
    
    def loads(self, data: bytes) -> Any:
        """Load from bytes."""
        if not isinstance(data, bytes):
            raise TypeError(f"Expected bytes, got {type(data).__name__}")
        
        # Use the from_bytes method to create KGraphSerialized instance
        serialized = KGraphSerialized.from_bytes(data)
        # Get bytes back and convert to object
        json_bytes = serialized.get_data()
        json_str = json_bytes.decode('utf-8')
        return json.loads(json_str)
    
    def dumps_typed(self, obj: Any) -> KGraphSerialized:
        """Serialize object and return KGraphSerialized container."""
        # Do the smart conversion work first
        json_serializable_data = self._make_json_serializable(obj)
        # Convert to bytes
        json_bytes = json.dumps(json_serializable_data, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        # Then let KGraphSerialized handle pure encoding
        return KGraphSerialized(data=json_bytes, original_type=type(obj).__name__)
    
    def loads_typed(self, data: Any) -> Any:
        """Deserialize from KGraphSerialized container."""
        if not isinstance(data, KGraphSerialized):
            raise TypeError(f"Expected KGraphSerialized object, got {type(data).__name__}")
        # Get bytes back and convert to object
        json_bytes = data.get_data()
        json_str = json_bytes.decode('utf-8')
        return json.loads(json_str)
    
