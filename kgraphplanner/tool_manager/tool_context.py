from typing import Optional
from typing_extensions import TypedDict


# accessed by a tool via the tool_manager
# used as needed for handling tool requests
class ToolContext(TypedDict):
    tool_person_name: str
    tool_login_name: str
    tool_account_uri: str
    tool_login_uri: str

    # may be passed through to tool server for authorization
    tool_jwt_string: Optional[str]

