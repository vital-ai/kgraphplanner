import uuid
from typing import Dict
from kgraphplanner.vital_agent_rest_resource_client.tools.place_search.place_search_response import \
    PlaceSearchResponse, PlaceDetails, PlaceSearchData
from kgraphplanner.vital_agent_rest_resource_client.tools.tool_handler import ToolHandler
from kgraphplanner.vital_agent_rest_resource_client.tools.tool_parameters import ToolParameters
from kgraphplanner.vital_agent_rest_resource_client.tools.tool_results import ToolResults

class PlaceSearchToolHandler(ToolHandler):

    def handle_response(self, tool_parameters: ToolParameters, response_json: dict) -> ToolResults:

        place_search_data = self.parse_place_search_response(tool_parameters, response_json)

        place_search_response = PlaceSearchResponse(
            place_search_data=place_search_data
        )

        return place_search_response

    def parse_place_search_response(self, tool_parameters: ToolParameters, response_json: Dict) -> PlaceSearchData:

        place_search_results = response_json.get('place_search_results', [])

        place_details_list = []

        for place in place_search_results:
            place_details = PlaceDetails(
                name=place.get('name', ''),
                address=place.get('address', ''),
                place_id=place.get('place_id', ''),
                latitude=place.get('latitude'),
                longitude=place.get('longitude'),
                business_status=place.get('business_status'),
                icon=place.get('icon'),
                types=place.get('types'),
                url=place.get('url'),
                vicinity=place.get('vicinity'),
                formatted_phone_number=place.get('formatted_phone_number'),
                website=place.get('website')
            )
            place_details_list.append(place_details)

        guid = uuid.uuid4()

        place_search_data = PlaceSearchData(
            tool_request_guid=str(guid),
            tool_data_class="PlaceSearchData",
            tool_name='place_search_tool',
            tool_parameters=tool_parameters,
            place_details_list=place_details_list
        )

        return place_search_data
