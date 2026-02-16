from typing import Optional, List
from typing_extensions import TypedDict
from kgraphplanner.vital_agent_rest_resource_client.tools.tool_data import ToolData
from kgraphplanner.vital_agent_rest_resource_client.tools.tool_results import ToolResults


class PlaceDetails(TypedDict):
    """
        Represents the place details

        Attributes:
            name (str): name of place
            address (str): address of place
            place_id (str): id of place
            latitude (float): latitude of place
            longitude (float): longitude of place
            business_status (str): business status of place
            icon (str): icon of place
            types (List[str]): types of place
            url (str): url of place
            vicinity (str): vicinity of place
            formatted_phone_number (str): formatted_phone_number of place
            website (str): website of place
    """

    name: str
    address: str
    place_id: str
    latitude: Optional[float]
    longitude: Optional[float]
    business_status: Optional[str]
    icon: Optional[str]
    types: Optional[List[str]]
    url: Optional[str]
    vicinity: Optional[str]
    formatted_phone_number: Optional[str]
    website: Optional[str]


class PlaceSearchData(ToolData):
    """
    Represents a list of place search results.

    Attributes:
        place_details_list List[PlaceDetails]: List of Place Details
    """

    place_details_list: List[PlaceDetails]


class PlaceSearchResponse(ToolResults):
    """
        Represents a list of place search results.

        Attributes:
            place_search_data (PlaceSearchData]: Place Search Data
    """

    place_search_data: PlaceSearchData
