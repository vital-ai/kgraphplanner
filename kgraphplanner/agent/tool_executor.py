import logging
from typing import Any, Callable, Sequence, Union
from langchain_core.load.serializable import Serializable
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.tools import tool as create_tool
from langgraph.utils.runnable import RunnableCallable

INVALID_TOOL_MSG_TEMPLATE = (
    "{requested_tool_name} is not a valid tool, "
    "try one of [{available_tool_names_str}]."
)


class ToolInvocationInterface:
    tool: str
    tool_input: Union[str, dict]


class ToolInvocation(Serializable):
    tool: str
    tool_input: Union[str, dict]


class ToolExecutor(RunnableCallable):

    def __init__(
        self,
        tools: Sequence[Union[BaseTool, Callable]],
        *,
        invalid_tool_msg_template: str = INVALID_TOOL_MSG_TEMPLATE,
    ) -> None:
        super().__init__(self._execute, afunc=self._aexecute, trace=False)
        tools_ = [
            tool if isinstance(tool, BaseTool) else create_tool(tool) for tool in tools
        ]
        self.tools = tools_
        self.tool_map = {t.name: t for t in tools}
        self.invalid_tool_msg_template = invalid_tool_msg_template

    def _execute(
        self, tool_invocation: ToolInvocationInterface, config: RunnableConfig
    ) -> Any:
        logger = logging.getLogger("HaleyAgentLogger")
        logger.info(f"Tool Execute Invoked")
        if tool_invocation.tool not in self.tool_map:
            return self.invalid_tool_msg_template.format(
                requested_tool_name=tool_invocation.tool,
                available_tool_names_str=", ".join([t.name for t in self.tools]),
            )
        else:
            tool = self.tool_map[tool_invocation.tool]
            output = tool.invoke(tool_invocation.tool_input, config)
            return output

    async def _aexecute(
        self, tool_invocation: ToolInvocationInterface, config: RunnableConfig
    ) -> Any:
        logger = logging.getLogger("HaleyAgentLogger")
        logger.info(f"Async Tool Execute Invoked")
        if tool_invocation.tool not in self.tool_map:
            return self.invalid_tool_msg_template.format(
                requested_tool_name=tool_invocation.tool,
                available_tool_names_str=", ".join([t.name for t in self.tools]),
            )
        else:
            tool = self.tool_map[tool_invocation.tool]

            logger.info(f"Async Tool Execute Invoked Input: {tool_invocation.tool_input}")
            output = await tool.ainvoke(tool_invocation.tool_input, config)
            return output
