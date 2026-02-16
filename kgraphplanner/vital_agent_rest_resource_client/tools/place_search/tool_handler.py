from typing import Dict
from .models import PlaceSearchOutput, PlaceDetails
from kgraphplanner.vital_agent_rest_resource_client.tools.tool_handler import ToolHandler
from kgraphplanner.vital_agent_rest_resource_client.tools.tool_parameters import ToolParameters
from kgraphplanner.vital_agent_rest_resource_client.tools.tool_results import ToolResults

class PlaceSearchToolHandler(ToolHandler):

    def handle_response(self, tool_parameters: ToolParameters, response_json: dict) -> ToolResults:

        place_search_output = self.parse_place_search_response(response_json)

        return place_search_output

    def parse_place_search_response(self, response_json: Dict) -> PlaceSearchOutput:

        # Extract results from tool_output structure
        place_search_results = response_json['tool_output'].get('results', [])

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

        place_search_output = PlaceSearchOutput(
            tool="place_search_tool",
            results=place_details_list
        )

        return place_search_output
