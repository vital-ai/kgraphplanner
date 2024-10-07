from kgraphplanner.tool_manager.abstract_tool import AbstractTool
from kgraphplanner.tool_manager.tool_cache import ToolCache


class ToolManager:
    def __init__(self, config: dict):
        self.tools = {}
        self.load_tools(config)
        self.tool_cache = ToolCache()

    def get_tool_cache(self) -> ToolCache:
        return self.tool_cache

    def load_tools(self, config):

        tool_config = config.get('tools', {})

        for tool_name, tool_info in tool_config.items():
            tool_class = tool_info['class']
            tool_config = tool_info.get('config', {})
            # Instantiate the tool class with its configuration
            self.tools[tool_name] = tool_class(tool_config)

    def add_tool(self, tool: AbstractTool):
        self.tools[tool.get_tool_name()] = tool

    def get_tool(self, tool_name) -> AbstractTool:
        tool = self.tools.get(tool_name, None)
        return tool

    def get_tool_list(self) -> list[AbstractTool]:
        tool_list = []
        for tool_name, tool in self.tools.items():
            tool_list.append(tool)
        return tool_list

    def call_tool(self, tool_name, request):
        tool = self.tools.get(tool_name)
        if tool:
            return tool.handle_request(request)
        else:
            return f"No tool found with the name: {tool_name}"


# provides index of tools to allow finding list of relevant tools


