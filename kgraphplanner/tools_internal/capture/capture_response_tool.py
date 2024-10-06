import pprint
from typing import Callable, Tuple

from kgraphplanner.structured_response.structured_response import StructuredResponse
from kgraphplanner.tool_manager.abstract_tool import AbstractTool
from kgraphplanner.tool_manager.tool_request import ToolRequest
from kgraphplanner.tool_manager.tool_response import ToolResponse
from langchain_core.tools import tool


class CaptureResponseTool(AbstractTool):
    def handle_request(self, request: ToolRequest) -> ToolResponse:
        pass

    def get_sample_text(self) -> str:
        pass

    def get_tool_function(self) -> Callable:

        @tool
        def capture_response(response_class_name: str, tool_response_guid) -> bool:
            """
            Use this to capture a structured tool response.
            :param response_class_name: The name of the response class, which should be a subclass of TypedDict
            :param tool_response_guid: The GUID of the response
            """

            # :param structured_tool_response: The structured response to be captured

            pp = pprint.PrettyPrinter(indent=4, width=40)

            print(f"CaptureResponseTool called with response_class_name: {response_class_name}")

            # print(f"StructuredResponse: {structured_tool_response}")
            # pp.pprint(structured_tool_response)

            print(f"CaptureResponseTool called with tool_response_guid: {tool_response_guid}")

            return True

        return capture_response

