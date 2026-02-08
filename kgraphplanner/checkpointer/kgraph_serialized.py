import logging
import base64
from datetime import datetime, timezone
from typing import Any, Dict

# Set up logger
logger = logging.getLogger(__name__)


class KGraphSerialized:
    """
    Data container for serialized data with base64 encoding/decoding.
    Handles the base64 JSON encoding internally.
    Implements pickle protocol for serialization.
    """
    
    def __init__(self, data: bytes, original_type: str = None, timestamp: str = None):
        """
        Initialize with pre-processed data to be encoded.
        
        Args:
            data: Pre-processed data as bytes to encrypt/encode
            original_type: The original type name of the data
            timestamp: Serialization timestamp
        """
        self.original_type = original_type or "bytes"
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()
        
        # Encode the data (pure container - just encrypt/encode)
        self.encoded_data = self._encode_data(data)
        
        logger.debug(f"KGraphSerialized container created for type {self.original_type}")
    
    def _encode_data(self, data: bytes) -> str:
        """Pure encoding - encrypt/encode already processed bytes."""
        return base64.b64encode(data).decode('utf-8')
    
    def get_data(self) -> bytes:
        """Decode and return the original bytes."""
        try:
            return base64.b64decode(self.encoded_data.encode('utf-8'))
        except Exception as e:
            logger.error(f"Failed to decode KGraphSerialized data: {e}")
            raise
    
    def __getstate__(self) -> Dict[str, Any]:
        """Return state for pickling."""
        return {
            'encoded_data': self.encoded_data,
            'original_type': self.original_type,
            'timestamp': self.timestamp
        }
    
    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore state from pickling."""
        self.encoded_data = state['encoded_data']
        self.original_type = state['original_type']
        self.timestamp = state['timestamp']
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'KGraphSerialized':
        """
        Create KGraphSerialized instance from bytes containing base64 encoded data.
        
        Args:
            data: Bytes containing base64-encoded JSON string
            
        Returns:
            KGraphSerialized instance with the encoded data
        """
        try:
            encoded_data = data.decode('utf-8')
            
            # Create empty instance and set the encoded data directly
            instance = cls.__new__(cls)
            instance.encoded_data = encoded_data
            instance.original_type = "unknown"  # We don't know the original type from just bytes
            instance.timestamp = datetime.now(timezone.utc).isoformat()
            
            logger.debug(f"KGraphSerialized container created from bytes")
            return instance
        except Exception as e:
            logger.error(f"Failed to create KGraphSerialized from bytes: {e}")
            raise
    
    def __repr__(self) -> str:
        return f"KGraphSerialized(type={self.original_type}, timestamp={self.timestamp})"
