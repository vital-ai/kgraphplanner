from kgraphplanner.tool_manager.abstract_tool import AbstractTool


class ToolManager:
    def __init__(self, config):
        self.tools = {}
        self.load_tools(config)

    def load_tools(self, config):
        for tool_name, tool_info in config.items():
            tool_class = tool_info['class']
            tool_config = tool_info.get('config', {})
            # Instantiate the tool class with its configuration
            self.tools[tool_name] = tool_class(tool_config)

    def get_tool(self, tool_name) -> AbstractTool:

        tool = self.tools.get(tool_name)

        if tool:
            return tool

        return None

    def add_tool(self, tool: AbstractTool):
        self.tools[tool.get_tool_name()] = tool

    def call_tool(self, tool_name, request):
        tool = self.tools.get(tool_name)
        if tool:
            return tool.handle_request(request)
        else:
            return f"No tool found with the name: {tool_name}"


# provides index of tools to allow finding list of relevant tools


