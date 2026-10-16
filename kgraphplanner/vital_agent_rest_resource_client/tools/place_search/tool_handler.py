from typing import Dict
from .models import PlaceSearchOutput, PlaceDetails
from kgraphplanner.vital_agent_rest_resource_client.tools.tool_handler import ToolHandler, build_model
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

        # Every field the model declares comes through; only the required
        # strings get a default.
        for place in place_search_results:
            data = dict(place)
            for key in ('name', 'address', 'place_id'):
                data[key] = data.get(key) or ''
            place_details_list.append(build_model(PlaceDetails, data))

        place_search_output = PlaceSearchOutput(
            tool="place_search_tool",
            results=place_details_list
        )

        return place_search_output
