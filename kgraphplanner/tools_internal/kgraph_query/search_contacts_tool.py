from typing import Callable, Tuple, List
from langchain_core.tools import tool
from kgraphplanner.tool_manager.abstract_tool import AbstractTool
from kgraphplanner.tool_manager.tool_request import ToolRequest
from kgraphplanner.tool_manager.tool_response import ToolResponse
from typing_extensions import TypedDict

# Note: this is meant to be implemented via a knowledge graph query of the user's knowledge graph

class Contact(TypedDict):
    """
    Contact is a dictionary that represents a contact from a contact list

    Attributes:
        contact_name (str): Name of Contact
        contact_telephone(str): Telephone number of Contact
        contact_email (str):  EMail of Contact
    """

    contact_name: str  # Name of Contact
    contact_telephone: str  # Telephone number of Contact
    contact_email: str  # EMail of Contact


class SearchContactsTool(AbstractTool):
    def handle_request(self, tool_request: ToolRequest) -> ToolResponse:

        contact_name = tool_request.get_parameter('contact_name')

        tool_response = ToolResponse()

        contact = Contact()
        contact['contact_name'] = "John Smith"
        contact['contact_telephone'] = "555-555-5555"
        contact['contact_email'] = "john@example.com"

        contact_list = [contact]

        tool_response.add_parameter("results", contact_list)

        return tool_response

    def get_sample_text(self) -> str:
        pass

    def get_tool_function(self) -> Callable:

        @tool
        def search_contacts(contact_name: str) -> List[Contact]:
            """
            Use this search a contact list for a name of a person.
            Returns a list of matching contacts or an empty list if no matching contacts are found.

            :param contact_name: Name of Contact
            :type contact_name: str
            :return: List of matching contacts or an empty list if no matching contacts are found.
            :rtype: List[Contact]
            """

            params = {contact_name: contact_name}

            tool_request = ToolRequest(params)

            tool_response = self.handle_request(tool_request)

            contact_list: list[Contact] = tool_response.get_parameter("results")

            return contact_list

        return search_contacts

