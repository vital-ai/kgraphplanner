from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class ToolParameters(BaseModel):
    """
    Base class for tool-specific input parameters.
    This will be extended by specific tool parameter classes.
    """
    
    model_config = {"extra": "allow"}  # Allow additional fields for tool-specific parameters
        
    def model_dump_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()
