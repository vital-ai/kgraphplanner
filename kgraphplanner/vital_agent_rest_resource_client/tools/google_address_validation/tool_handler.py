from typing import Dict
from .models import AddressValidationOutput, AddressValidationResult, AddressComponent
from kgraphplanner.vital_agent_rest_resource_client.tools.tool_handler import ToolHandler
from kgraphplanner.vital_agent_rest_resource_client.tools.tool_parameters import ToolParameters
from kgraphplanner.vital_agent_rest_resource_client.tools.tool_results import ToolResults

class AddressValidationToolHandler(ToolHandler):

    def handle_response(self, tool_parameters: ToolParameters, response_json: dict) -> ToolResults:

        address_validation_output = self.parse_address_validation_response(response_json)

        return address_validation_output

    def parse_address_validation_response(self, response_json: Dict) -> AddressValidationOutput:

        # Extract results from tool_output structure
        validation_results = response_json['tool_output'].get('results', [])

        address_validation_results = []

        for result in validation_results:
            # Parse address components
            address_components = []
            components_data = result.get('address_components', [])
            
            for component in components_data:
                address_component = AddressComponent(
                    component_name=component.get('component_name', ''),
                    component_type=component.get('component_type', ''),
                    confirmation_level=component.get('confirmation_level', '')
                )
                address_components.append(address_component)

            # Create address validation result
            address_result = AddressValidationResult(
                formatted_address=result.get('formatted_address', ''),
                postal_address=result.get('postal_address', {}),
                address_components=address_components,
                geocode=result.get('geocode'),
                metadata=result.get('metadata'),
                usps_data=result.get('usps_data')
            )
            address_validation_results.append(address_result)

        address_validation_output = AddressValidationOutput(
            tool="google_address_validation_tool",
            results=address_validation_results
        )

        return address_validation_output
