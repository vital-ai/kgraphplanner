from kgraphplanner.tool_manager.abstract_tool import AbstractTool
from typing import Callable, Tuple
from langchain_core.tools import tool
from kgraphplanner.tool_manager.tool_request import ToolRequest
from kgraphplanner.tool_manager.tool_response import ToolResponse
from typing_extensions import TypedDict


class SMSMessage(TypedDict):
    """
    SMSMessage is a dictionary that represents an SMS Message to send

    Attributes:
        recipient_telephone (str): Telephone number of Recipient
        recipient_name (str): Name of recipient

        sender_telephone (str): Telephone number of Sender
        sender_name (str): Name of sender
        message (str):  Content of the SMS Message
    """

    recipient_telephone: str  # Telephone number of Recipient
    recipient_name: str  # Name of recipient

    sender_telephone: str  # Telephone number of Sender
    sender_name: str  # Name of sender
    message: str  # Content of the SMS Message


class SendMessageTool(AbstractTool):
    def handle_request(self, request: ToolRequest) -> ToolResponse:
        pass

    def get_sample_text(self) -> str:
        pass

    def get_tool_function(self) -> Callable:

        @tool
        def send_sms_message(sms_message: SMSMessage) -> bool:
            """
            Use this tool to send an SMS Message.

            :param sms_message: The SMSMessage object that contains the message details.
            :type sms_message: SMSMessage
            :returns True if successful, False otherwise.
            :rtype: bool
            """

            return True

        return send_sms_message


