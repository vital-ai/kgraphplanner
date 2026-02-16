from pydantic import BaseModel, Field
from typing import List, Optional, Literal


class PlaceDetails(BaseModel):
    name: str = Field(..., description="Name of the place")
    address: str = Field(..., description="Formatted address")
    place_id: str = Field(..., description="Google Places ID")
    latitude: Optional[float] = Field(None, description="Latitude coordinate")
    longitude: Optional[float] = Field(None, description="Longitude coordinate")
    business_status: Optional[str] = Field(None, description="Business operational status")
    icon: Optional[str] = Field(None, description="Icon URL")
    types: Optional[List[str]] = Field(None, description="Place types")
    url: Optional[str] = Field(None, description="Google Maps URL")
    vicinity: Optional[str] = Field(None, description="Vicinity description")
    formatted_phone_number: Optional[str] = Field(None, description="Formatted phone number")
    website: Optional[str] = Field(None, description="Website URL")


class PlaceSearchInput(BaseModel):
    """Input model for Place Search tool"""
    place_search_string: str = Field(..., description="Search query for places", min_length=1)

    model_config = {
        "json_schema_extra": {
            "example": {
                "place_search_string": "restaurants near me"
            }
        }
    }


class PlaceSearchOutput(BaseModel):
    """Output model for Place Search tool"""
    tool: Literal["place_search_tool"] = Field(..., description="Tool identifier")
    results: List[PlaceDetails] = Field(default_factory=list, description="Place search results")

    model_config = {
        "json_schema_extra": {
            "example": {
                "tool": "place_search_tool",
                "results": [
                    {
                        "name": "Example Restaurant",
                        "address": "123 Main St, City, State",
                        "place_id": "ChIJexample123",
                        "latitude": 40.7128,
                        "longitude": -74.0060,
                        "business_status": "OPERATIONAL",
                        "types": ["restaurant", "food"],
                        "formatted_phone_number": "(555) 123-4567"
                    }
                ]
            }
        }
    }


