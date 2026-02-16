from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal


class WeatherData(BaseModel):
    latitude: float = Field(..., description="Latitude coordinate")
    longitude: float = Field(..., description="Longitude coordinate")
    timezone: str = Field(..., description="Timezone information")
    current: Optional[Dict[str, Any]] = Field(None, description="Current weather conditions")
    daily: Optional[Dict[str, Any]] = Field(None, description="Daily weather forecast")
    hourly: Optional[Dict[str, Any]] = Field(None, description="Hourly weather forecast")


class WeatherInput(BaseModel):
    """Input model for Weather tool"""
    latitude: float = Field(..., description="Latitude coordinate", ge=-90, le=90)
    longitude: float = Field(..., description="Longitude coordinate", ge=-180, le=180)
    include_previous: Optional[bool] = Field(False, description="Include previous 10 days of data")
    use_archive: Optional[bool] = Field(False, description="Use archive weather data")
    archive_date: Optional[str] = Field(None, description="Archive date in YYYY-MM-DD format")

    model_config = {
        "json_schema_extra": {
            "example": {
                "latitude": 40.7128,
                "longitude": -74.0060,
                "include_previous": False,
                "use_archive": False
            }
        }
    }


class WeatherOutput(BaseModel):
    """Output model for Weather tool"""
    tool: Literal["weather_tool"] = Field(..., description="Tool identifier")
    weather_data: WeatherData = Field(..., description="Weather information")

    model_config = {
        "json_schema_extra": {
            "example": {
                "tool": "weather_tool",
                "weather_data": {
                    "latitude": 40.7128,
                    "longitude": -74.0060,
                    "timezone": "America/New_York",
                    "current": {
                        "temperature_2m": 72.5,
                        "weather_code": 0,
                        "wind_speed_10m": 5.2
                    },
                    "daily": {
                        "temperature_2m_max": [75.2, 78.1],
                        "temperature_2m_min": [65.3, 68.7]
                    }
                }
            }
        }
    }


