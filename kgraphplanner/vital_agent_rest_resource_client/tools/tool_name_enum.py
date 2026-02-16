"""
Tool name enumeration for available tools.
"""

from enum import Enum


class ToolName(str, Enum):
    """Available tool names"""
    google_address_validation_tool = "google_address_validation_tool"
    google_web_search_tool = "google_web_search_tool"
    place_search_tool = "place_search_tool"
    weather_tool = "weather_tool"
