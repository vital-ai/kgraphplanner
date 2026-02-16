from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class ToolResults(BaseModel):
    """
    Base class for tool execution results.
    This will be extended by specific tool result classes.
    """
    
    model_config = {"extra": "allow"}  # Allow additional fields for tool-specific results
        
    def model_dump_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()


