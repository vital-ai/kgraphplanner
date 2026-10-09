import logging
from typing import Callable, Type

logger = logging.getLogger(__name__)

from pydantic import BaseModel
from langchain_core.tools import tool, ToolException

from kgraphplanner.tools.tool_errors import fail, reports_failures

from kgraphplanner.vital_agent_rest_resource_client.tools.google_address_validation.models import (
    AddressValidationInput, 
    AddressValidationOutput
)
from kgraphplanner.vital_agent_rest_resource_client.vital_agent_rest_resource_client import (
    VitalAgentRestResourceClient
)
from kgraphplanner.vital_agent_rest_resource_client.tools.tool_name_enum import ToolName as ToolNameEnum

from kgraphplanner.tool_manager.tool_inf import AbstractTool
from kgraphplanner.tools.tool_logging import describe_error, logged


class AddressValidationTool(AbstractTool):
    """Address validation tool implementation using VitalAgentRestResourceClient."""
    
    def __init__(self, config, tool_manager=None):
        super().__init__(
            config=config,
            tool_manager=tool_manager,
            name=ToolNameEnum.google_address_validation_tool.value,
            description="Validate and standardize postal addresses"
        )
    
    def get_tool_schema(self) -> Type[BaseModel]:
        """Get the Pydantic schema for address validation parameters."""
        return AddressValidationInput
    
    def get_tool_function(self) -> Callable:
        """Get the tool function for address validation."""
        
        @tool(args_schema=AddressValidationInput)
        async def google_address_validation_tool(address: str) -> AddressValidationOutput:
            """
            Validate and standardize postal addresses.
            
            Args:
                address: The address string to validate
                
            Returns:
                AddressValidationOutput: Address validation results
            """
            
            # Create AddressValidationInput from parameter
            address_validation_input = AddressValidationInput(address=address)
            
            # Get tool endpoint from config
            tool_endpoint = self.config.get("tool_endpoint")
            if not tool_endpoint:
                fail("Address validation is unavailable: tool endpoint is not configured.")
            
            # Create client configuration
            client_config = {
                "tool_endpoint": tool_endpoint,
                "log_content": self.log_content,
            }
            
            # Get JWT token from tool manager if available
            jwt_token = None
            if self.tool_manager:
                jwt_token = self.tool_manager.get_jwt_token()
            
            # Initialize the client
            client = VitalAgentRestResourceClient(client_config, jwt_token)
            
            try:
                # Execute the validation (async)
                tool_response = await client.handle_tool_request(ToolNameEnum.google_address_validation_tool.value, address_validation_input)
                
                if not tool_response.success or tool_response.tool_output is None:
                    detail = tool_response.error_message
                    logger.error(f"Address validation tool failed: {logged(detail, self.log_content, label='error body')}")
                    fail(f"Address validation is unavailable"
                         + (f": {detail}" if detail else "."))
                
                return tool_response.tool_output
                
            except ToolException:
                raise
            except Exception as e:
                logger.error(f"Address validation tool error: {describe_error(e, log_content=self.log_content)}")
                fail(f"Address validation is unavailable (error: {type(e).__name__}: {e})")
        
        return reports_failures(google_address_validation_tool)
    
    def validate_address(self, address: str) -> AddressValidationOutput:
        """
        Direct execution method for address validation.
        
        Args:
            address: Address string to validate
            
        Returns:
            AddressValidationOutput: Validation results
        """
        # Create input model
        address_input = AddressValidationInput(
            address=address
        )
        
        tool_function = self.get_tool_function()
        return tool_function.invoke(address_input)
