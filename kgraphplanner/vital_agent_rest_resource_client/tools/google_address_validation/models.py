from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal


class AddressComponent(BaseModel):
    component_name: str = Field(..., description="Component name text")
    component_type: str = Field(..., description="Type of address component")
    confirmation_level: str = Field(..., description="Confidence level of the component")


class AddressValidationResult(BaseModel):
    formatted_address: str = Field(..., description="Standardized formatted address")
    postal_address: Dict[str, Any] = Field(default_factory=dict, description="Postal address details")
    address_components: List[AddressComponent] = Field(default_factory=list, description="Individual address components")
    geocode: Optional[Dict[str, Any]] = Field(None, description="Geocoding information")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    usps_data: Optional[Dict[str, Any]] = Field(None, description="USPS validation data")


class AddressValidationInput(BaseModel):
    """Input model for Google Address Validation tool"""
    address: str = Field(..., description="Address to validate", min_length=1)

    model_config = {
        "json_schema_extra": {
            "example": {
                "address": "1600 Amphitheatre Parkway, Mountain View, CA"
            }
        }
    }


class AddressValidationOutput(BaseModel):
    """Output model for Google Address Validation tool"""
    tool: Literal["google_address_validation_tool"] = Field(..., description="Tool identifier")
    results: List[AddressValidationResult] = Field(default_factory=list, description="Address validation results")

    model_config = {
        "json_schema_extra": {
            "example": {
                "tool": "google_address_validation_tool",
                "results": [
                    {
                        "formatted_address": "1600 Amphitheatre Pkwy, Mountain View, CA 94043, USA",
                        "postal_address": {},
                        "address_components": [],
                        "geocode": {},
                        "metadata": {},
                        "usps_data": {}
                    }
                ]
            }
        }
    }


