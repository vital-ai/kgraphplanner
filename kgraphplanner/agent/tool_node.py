import asyncio
import json
import queue
from copy import copy
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

from langchain_core.messages import AIMessage, AnyMessage, ToolCall, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import get_config_list, get_executor_for_config
from langchain_core.tools import BaseTool, InjectedToolArg
from langchain_core.tools import tool as create_tool
from langgraph.utils.runnable import RunnableCallable
from typing_extensions import get_args


INVALID_TOOL_NAME_ERROR_TEMPLATE = (
    "Error: {requested_tool} is not a valid tool, try one of [{available_tools}]."
)
TOOL_CALL_ERROR_TEMPLATE = "Error: {error}\n Please fix your mistakes."


def str_output(output: Any) -> str:
    if isinstance(output, str):
        return output
    else:
        try:
            return json.dumps(output)
        except Exception:
            return str(output)


class ToolNode(RunnableCallable):

    def __init__(
        self,
        tools: Sequence[Union[BaseTool, Callable]],
        *,
        name: str = "tools",
        tags: Optional[list[str]] = None,
        handle_tool_errors: Optional[bool] = True,
        reasoning_queue: queue.Queue = None
    ) -> None:
        super().__init__(self._func, self._afunc, name=name, tags=tags, trace=False)
        self.tools_by_name: Dict[str, BaseTool] = {}
        self.handle_tool_errors = handle_tool_errors
        self.reasoning_queue = reasoning_queue
        for tool_ in tools:
            if not isinstance(tool_, BaseTool):
                tool_ = create_tool(tool_)
            self.tools_by_name[tool_.name] = tool_

    def _func(
        self, input: Union[list[AnyMessage], dict[str, Any]], config: RunnableConfig
    ) -> Any:
        tool_calls, output_type = self._parse_input(input)
        config_list = get_config_list(config, len(tool_calls))
        with get_executor_for_config(config) as executor:
            outputs = [*executor.map(self._run_one, tool_calls, config_list)]
        return outputs if output_type == "list" else {"messages": outputs}

    async def _afunc(
        self, input: Union[list[AnyMessage], dict[str, Any]], config: RunnableConfig
    ) -> Any:
        tool_calls, output_type = self._parse_input(input)
        outputs = await asyncio.gather(
            *(self._arun_one(call, config) for call in tool_calls)
        )
        return outputs if output_type == "list" else {"messages": outputs}

    def _run_one(self, call: ToolCall, config: RunnableConfig) -> ToolMessage:
        if invalid_tool_message := self._validate_tool_call(call):
            return invalid_tool_message

        try:
            input = {**call, **{"type": "tool_call"}}

            print(f"Tool Call Invoking: {self.tools_by_name[call['name']]}")
            # print(f"Tool Call Config: {config}")

            tool_name = call['name']

            reasoning_message = {
                "agent_thought": f"calling tool: {tool_name}"
            }

            if self.reasoning_queue:
                self.reasoning_queue.put(reasoning_message)

            tool_message: ToolMessage = self.tools_by_name[call["name"]].invoke(
                input, config
            )

            reasoning_message = {
                "agent_thought": f"completed calling tool: {tool_name}"
            }

            if self.reasoning_queue:
                self.reasoning_queue.put(reasoning_message)

            # TODO: handle this properly in core
            tool_message.content = str_output(tool_message.content)
            return tool_message
        except Exception as e:
            if not self.handle_tool_errors:
                raise e
            content = TOOL_CALL_ERROR_TEMPLATE.format(error=repr(e))
            return ToolMessage(content, name=call["name"], tool_call_id=call["id"])

    async def _arun_one(self, call: ToolCall, config: RunnableConfig) -> ToolMessage:
        if invalid_tool_message := self._validate_tool_call(call):
            return invalid_tool_message
        try:
            input = {**call, **{"type": "tool_call"}}

            print(f"Async Tool Call Invoking: {self.tools_by_name[call['name']]}")

            tool_message: ToolMessage = await self.tools_by_name[call["name"]].ainvoke(
                input, config
            )
            # TODO: handle this properly in core
            tool_message.content = str_output(tool_message.content)
            return tool_message
        except Exception as e:
            if not self.handle_tool_errors:
                raise e
            content = TOOL_CALL_ERROR_TEMPLATE.format(error=repr(e))
            return ToolMessage(content, name=call["name"], tool_call_id=call["id"])

    def _parse_input(
        self, input: Union[list[AnyMessage], dict[str, Any]]
    ) -> Tuple[List[ToolCall], Literal["list", "dict"]]:
        if isinstance(input, list):
            output_type = "list"
            message: AnyMessage = input[-1]
        elif messages := input.get("messages", []):
            output_type = "dict"
            message = messages[-1]
        else:
            raise ValueError("No message found in input")

        if not isinstance(message, AIMessage):
            raise ValueError("Last message is not an AIMessage")

        tool_calls = [
            self._inject_state(call, input)
            for call in cast(AIMessage, message).tool_calls
        ]
        return tool_calls, output_type

    def _validate_tool_call(self, call: ToolCall) -> Optional[ToolMessage]:
        if (requested_tool := call["name"]) not in self.tools_by_name:
            content = INVALID_TOOL_NAME_ERROR_TEMPLATE.format(
                requested_tool=requested_tool,
                available_tools=", ".join(self.tools_by_name.keys()),
            )
            return ToolMessage(content, name=requested_tool, tool_call_id=call["id"])
        else:
            return None

    def _inject_state(
        self, tool_call: ToolCall, input: Union[list[AnyMessage], dict[str, Any]]
    ) -> ToolCall:
        if tool_call["name"] not in self.tools_by_name:
            return tool_call
        state_args = _get_state_args(self.tools_by_name[tool_call["name"]])
        if state_args and not isinstance(input, dict):
            required_fields = list(state_args.values())
            if (
                len(required_fields) == 1
                and required_fields[0] == "messages"
                or required_fields[0] is None
            ):
                input = {"messages": input}
            else:
                err_msg = (
                    f"Invalid input to ToolNode. Tool {tool_call['name']} requires "
                    f"graph state dict as input."
                )
                if any(state_field for state_field in state_args.values()):
                    required_fields_str = ", ".join(f for f in required_fields if f)
                    err_msg += f" State should contain fields {required_fields_str}."
                raise ValueError(err_msg)
        tool_call_copy: ToolCall = copy(tool_call)
        tool_call_copy["args"] = {
            **tool_call_copy["args"],
            **{
                tool_arg: cast(dict, input)[state_field] if state_field else input
                for tool_arg, state_field in state_args.items()
            },
        }
        return tool_call_copy


def tools_condition(
    state: Union[list[AnyMessage], dict[str, Any]],
) -> Literal["tools", "__end__"]:

    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "__end__"


class InjectedState(InjectedToolArg):

    def __init__(self, field: Optional[str] = None) -> None:
        self.field = field


def _get_state_args(tool: BaseTool) -> Dict[str, Optional[str]]:
    full_schema = tool.get_input_schema()
    tool_args_to_state_fields: Dict = {}
    for name, type_ in full_schema.__annotations__.items():
        injections = [
            type_arg
            for type_arg in get_args(type_)
            if isinstance(type_arg, InjectedState)
            or (isinstance(type_arg, type) and issubclass(type_arg, InjectedState))
        ]
        if len(injections) > 1:
            raise ValueError(
                "A tool argument should not be annotated with InjectedState more than "
                f"once. Received arg {name} with annotations {injections}."
            )
        elif len(injections) == 1:
            injection = injections[0]
            if isinstance(injection, InjectedState) and injection.field:
                tool_args_to_state_fields[name] = injection.field
            else:
                tool_args_to_state_fields[name] = None
        else:
            pass
    return tool_args_to_state_fields
