from abc import ABC, abstractmethod

from kgraphplanner.vital_agent_rest_resource_client.tools.tool_parameters import ToolParameters
from kgraphplanner.vital_agent_rest_resource_client.tools.tool_results import ToolResults


class ToolHandler(ABC):

    @abstractmethod
    def handle_response(self, tool_parameters: ToolParameters, response: dict) -> ToolResults:
        pass

