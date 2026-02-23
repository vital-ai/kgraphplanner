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

    def compact_dump(self) -> str:
        """Return a compact markdown text report instead of raw JSON."""
        wd = self.weather_data
        lines = [f"**Weather** (lat={wd.latitude}, lon={wd.longitude}, tz={wd.timezone})"]

        # Current conditions
        c = wd.current
        if c:
            temp = c.get("temperature_2m")
            feels = c.get("apparent_temperature")
            humidity = c.get("relative_humidity_2m")
            cloud = c.get("cloud_cover")
            precip = c.get("precipitation", 0)
            precip_prob = c.get("precipitation_probability")
            wind = c.get("wind_speed_10m")
            gusts = c.get("wind_gusts_10m")
            wind_dir = c.get("wind_direction_10m")
            code = c.get("weather_code")
            t = c.get("time", "")

            parts = []
            if temp is not None:
                parts.append(f"{temp}°F")
            if feels is not None:
                parts.append(f"feels {feels}°F")
            if humidity is not None:
                parts.append(f"humidity {humidity}%")
            if cloud is not None:
                parts.append(f"cloud {cloud}%")
            if precip_prob is not None:
                parts.append(f"precip_prob {precip_prob}%")
            if precip:
                parts.append(f"precip {precip}")
            if wind is not None:
                wind_str = f"wind {wind} m/s"
                if wind_dir is not None:
                    wind_str += f" dir {wind_dir}°"
                if gusts is not None:
                    wind_str += f" gusts {gusts}"
                parts.append(wind_str)
            if code is not None:
                parts.append(f"code {code}")

            lines.append(f"Current ({t}): {', '.join(parts)}")

        # Daily summary — today + next 2 days only
        d = wd.daily
        if d and "time" in d:
            days = d["time"]
            n = min(len(days), 3)
            for i in range(n):
                day_parts = [days[i]]
                hi = d.get("temperature_2m_max", [None]*n)
                lo = d.get("temperature_2m_min", [None]*n)
                pp = d.get("precipitation_probability_max", [None]*n)
                ps = d.get("precipitation_sum", [None]*n)
                wg = d.get("wind_gusts_10m_max", [None]*n)
                if i < len(hi) and hi[i] is not None:
                    day_parts.append(f"hi {hi[i]}°F")
                if i < len(lo) and lo[i] is not None:
                    day_parts.append(f"lo {lo[i]}°F")
                if i < len(pp) and pp[i] is not None:
                    day_parts.append(f"precip_prob {pp[i]}%")
                if i < len(ps) and ps[i] is not None and ps[i] > 0:
                    day_parts.append(f"precip {ps[i]}")
                if i < len(wg) and wg[i] is not None:
                    day_parts.append(f"gusts {wg[i]}")
                lines.append(f"Day {i}: {', '.join(day_parts)}")

        return "\n".join(lines)

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


