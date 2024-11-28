# from __future__ import annotations
import inspect
import asyncio
import json
import logging
from copy import copy
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
    get_type_hints,
)

from langchain_core.messages import AIMessage, AnyMessage, ToolCall, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.utils import Input
from langchain_core.tools.base import get_all_basemodel_annotations
from langgraph.errors import GraphInterrupt
from typing_extensions import Annotated, get_origin
from langchain_core.runnables.config import get_config_list, get_executor_for_config
from langchain_core.tools import BaseTool, InjectedToolArg
from langchain_core.tools import tool as create_tool
from typing_extensions import get_args
from langgraph.store.base import BaseStore
from langgraph.utils.runnable import RunnableCallable

# if TYPE_CHECKING:
#    from pydantic import BaseModel

from pydantic import BaseModel

INVALID_TOOL_NAME_ERROR_TEMPLATE = (
    "Error: {requested_tool} is not a valid tool, try one of [{available_tools}]."
)
TOOL_CALL_ERROR_TEMPLATE = "Error: {error}\n Please fix your mistakes."



def msg_content_output(output: Any) -> str | List[dict]:
    recognized_content_block_types = ("image", "image_url", "text", "json")
    if isinstance(output, str):
        return output
    elif all(
        [
            isinstance(x, dict) and x.get("type") in recognized_content_block_types
            for x in output
        ]
    ):
        return output
    # Technically a list of strings is also valid message content but it's not currently
    # well tested that all chat models support this. And for backwards compatibility
    # we want to make sure we don't break any existing ToolNode usage.
    else:
        try:
            return json.dumps(output, ensure_ascii=False)
        except Exception:
            return str(output)


def str_output(output: Any) -> str:
    if isinstance(output, str):
        return output
    else:
        try:
            return json.dumps(output)
        except Exception:
            return str(output)

def _handle_tool_error(
    e: Exception,
    *,
    flag: Union[
        bool,
        str,
        Callable[..., str],
        tuple[type[Exception], ...],
    ],
) -> str:
    if isinstance(flag, (bool, tuple)):
        content = TOOL_CALL_ERROR_TEMPLATE.format(error=repr(e))
    elif isinstance(flag, str):
        content = flag
    elif callable(flag):
        content = flag(e)
    else:
        raise ValueError(
            f"Got unexpected type of `handle_tool_error`. Expected bool, str "
            f"or callable. Received: {flag}"
        )
    return content


def _infer_handled_types(handler: Callable[..., str]) -> tuple[type[Exception]]:
    sig = inspect.signature(handler)
    params = list(sig.parameters.values())
    if params:
        # If it's a method, the first argument is typically 'self' or 'cls'
        if params[0].name in ["self", "cls"] and len(params) == 2:
            first_param = params[1]
        else:
            first_param = params[0]

        type_hints = get_type_hints(handler)
        if first_param.name in type_hints:
            origin = get_origin(first_param.annotation)
            if origin is Union:
                args = get_args(first_param.annotation)
                if all(issubclass(arg, Exception) for arg in args):
                    return tuple(args)
                else:
                    raise ValueError(
                        "All types in the error handler error annotation must be Exception types. "
                        "For example, `def custom_handler(e: Union[ValueError, TypeError])`. "
                        f"Got '{first_param.annotation}' instead."
                    )

            exception_type = type_hints[first_param.name]
            if Exception in exception_type.__mro__:
                return (exception_type,)
            else:
                raise ValueError(
                    f"Arbitrary types are not supported in the error handler signature. "
                    "Please annotate the error with either a specific Exception type or a union of Exception types. "
                    "For example, `def custom_handler(e: ValueError)` or `def custom_handler(e: Union[ValueError, TypeError])`. "
                    f"Got '{exception_type}' instead."
                )

    # If no type information is available, return (Exception,) for backwards compatibility.
    return (Exception,)


class ToolNode(RunnableCallable):

    def __init__(
        self,
        tools: Sequence[Union[BaseTool, Callable]],
        *,
        name: str = "tools",
        tags: Optional[list[str]] = None,
        handle_tool_errors: Union[
            bool, str, Callable[..., str], tuple[type[Exception], ...]
        ] = True,
        messages_key: str = "messages",
        reasoning_queue: asyncio.Queue = None
    ) -> None:
        super().__init__(self._func, self._afunc, name=name, tags=tags, trace=False)

        self.tools_by_name: Dict[str, BaseTool] = {}
        self.tool_to_state_args: Dict[str, Dict[str, Optional[str]]] = {}
        self.tool_to_store_arg: Dict[str, Optional[str]] = {}
        self.handle_tool_errors = handle_tool_errors
        self.messages_key = messages_key
        self.reasoning_queue = reasoning_queue

        for tool_ in tools:
            if not isinstance(tool_, BaseTool):
                tool_ = create_tool(tool_)
            self.tools_by_name[tool_.name] = tool_
            self.tool_to_state_args[tool_.name] = _get_state_args(tool_)
            self.tool_to_store_arg[tool_.name] = _get_store_arg(tool_)

    # def convert_input(self, input_dict):
    #     def recursive_convert(d):
    #         for key, value in d.items():
    #             if isinstance(value, str):
    #                 try:
    #                     parsed_value = json.loads(value)
    #                     if isinstance(parsed_value, dict):
    #                         if "request_class_name" in parsed_value:
    #                             d[key] = {"agent_request": parsed_value}
    #                         else:
    #                             d[key] = recursive_convert(parsed_value)
    #                 except json.JSONDecodeError:
    #                     pass
    #             elif isinstance(value, dict):
    #                 d[key] = recursive_convert(value)
    #         return d
    #
    #     return recursive_convert(input_dict)

    def _func(
            self,
            input: Union[
                list[AnyMessage],
                dict[str, Any],
                BaseModel,
            ],
            config: RunnableConfig,
            *,
            store: BaseStore,
    ) -> Any:
        tool_calls, output_type = self._parse_input(input, store)
        config_list = get_config_list(config, len(tool_calls))
        with get_executor_for_config(config) as executor:
            outputs = [*executor.map(self._run_one, tool_calls, config_list)]
        # TypedDict, pydantic, dataclass, etc. should all be able to load from dict
        return outputs if output_type == "list" else {self.messages_key: outputs}

    def invoke(
            self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Any:
        if "store" not in kwargs:
            kwargs["store"] = None
        return super().invoke(input, config, **kwargs)

    async def ainvoke(
            self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Any:
        if "store" not in kwargs:
            kwargs["store"] = None
        return await super().ainvoke(input, config, **kwargs)

    async def _afunc(
            self,
            input: Union[
                list[AnyMessage],
                dict[str, Any],
                BaseModel,
            ],
            config: RunnableConfig,
            *,
            store: BaseStore,
    ) -> Any:
        tool_calls, output_type = self._parse_input(input, store)
        outputs = await asyncio.gather(
            *(self._arun_one(call, config) for call in tool_calls)
        )
        # TypedDict, pydantic, dataclass, etc. should all be able to load from dict
        return outputs if output_type == "list" else {self.messages_key: outputs}

    def _run_one(self, call: ToolCall, config: RunnableConfig) -> ToolMessage:
        if invalid_tool_message := self._validate_tool_call(call):
            return invalid_tool_message

        try:
            input = {**call, **{"type": "tool_call"}}
            tool_message: ToolMessage = self.tools_by_name[call["name"]].invoke(
                input, config
            )
            tool_message.content = cast(
                Union[str, list], msg_content_output(tool_message.content)
            )
            return tool_message
        # GraphInterrupt is a special exception that will always be raised.
        # It can be triggered in the following scenarios:
        # (1) a NodeInterrupt is raised inside a tool
        # (2) a NodeInterrupt is raised inside a graph node for a graph called as a tool
        # (3) a GraphInterrupt is raised when a subgraph is interrupted inside a graph called as a tool
        # (2 and 3 can happen in a "supervisor w/ tools" multi-agent architecture)
        # switch to GraphBubble
        except GraphInterrupt as e:
            raise e
        except Exception as e:
            if isinstance(self.handle_tool_errors, tuple):
                handled_types: tuple = self.handle_tool_errors
            elif callable(self.handle_tool_errors):
                handled_types = _infer_handled_types(self.handle_tool_errors)
            else:
                # default behavior is catching all exceptions
                handled_types = (Exception,)

            # Unhandled
            if not self.handle_tool_errors or not isinstance(e, handled_types):
                raise e
            # Handled
            else:
                content = _handle_tool_error(e, flag=self.handle_tool_errors)

        return ToolMessage(
            content=content, name=call["name"], tool_call_id=call["id"], status="error"
        )

    def _run_one_before(self, call: ToolCall, config: RunnableConfig) -> ToolMessage:
        if invalid_tool_message := self._validate_tool_call(call):
            return invalid_tool_message

        try:
            input = {**call, **{"type": "tool_call"}}

            # input_updated = self.convert_input(input)

            print(f"Tool Call Invoking: {self.tools_by_name[call['name']]}")

            # print(f"Tool Call Config: {config}")

            print(f"Tool Call Invoking Input: {input}")

            input_tool_name = input.get("name", None)

            ############################################################
            # Temp work-around until tool calling schema issue resolved
            if input_tool_name == "CallAgentTool":
                input_args: dict =input.get("args", None)

                if input_args:
                    arg1_value = input_args.get("__arg1", None)

                    if arg1_value is not None and type(arg1_value) is str:
                        arg1_dict = json.loads(arg1_value)
                        input_args["__arg1"] = {"agent_request": arg1_dict}

                    if arg1_value is not None and type(arg1_value) is dict:
                        input_args["__arg1"] = {"agent_request": arg1_value}

            print(f"Tool Call Invoking Updated Input: {input}")

            ############################################################

            tool_name = call['name']

            reasoning_message = {
                "agent_thought": f"calling tool: {tool_name}"
            }

            # if self.reasoning_queue:
            #    await self.reasoning_queue.put(reasoning_message)

            tool_message: ToolMessage = self.tools_by_name[call["name"]].invoke(
                input, config
            )

            reasoning_message = {
                "agent_thought": f"completed calling tool: {tool_name}"
            }

            # if self.reasoning_queue:
            #   await  self.reasoning_queue.put(reasoning_message)

            # TODO: handle this properly in core
            tool_message.content = str_output(tool_message.content)
            return tool_message
        except Exception as e:
            if not self.handle_tool_errors:
                raise e
            content = TOOL_CALL_ERROR_TEMPLATE.format(error=repr(e))
            return ToolMessage(content, name=call["name"], tool_call_id=call["id"])

    async def _arun_one(self, call: ToolCall, config: RunnableConfig) -> ToolMessage:

        logger = logging.getLogger("HaleyAgentLogger")


        if invalid_tool_message := self._validate_tool_call(call):
            return invalid_tool_message

        try:
            input = {**call, **{"type": "tool_call"}}

            logger.info(f"Async Tool Call Invoking: {self.tools_by_name[call['name']]}")

            tool_name = call['name']

            reasoning_message = {
                "agent_thought": f"calling tool: {tool_name}"
            }

            if self.reasoning_queue:
                await self.reasoning_queue.put(reasoning_message)


            tool_message: ToolMessage = await self.tools_by_name[call["name"]].ainvoke(
                input, config
            )

            reasoning_message = {
                "agent_thought": f"completed calling tool: {tool_name}"
            }

            if self.reasoning_queue:
                await self.reasoning_queue.put(reasoning_message)

            tool_message.content = cast(
                Union[str, list], msg_content_output(tool_message.content)
            )
            return tool_message
        # GraphInterrupt is a special exception that will always be raised.
        # It can be triggered in the following scenarios:
        # (1) a NodeInterrupt is raised inside a tool
        # (2) a NodeInterrupt is raised inside a graph node for a graph called as a tool
        # (3) a GraphInterrupt is raised when a subgraph is interrupted inside a graph called as a tool
        # (2 and 3 can happen in a "supervisor w/ tools" multi-agent architecture)
        except GraphInterrupt as e:
            raise e
        except Exception as e:
            if isinstance(self.handle_tool_errors, tuple):
                handled_types: tuple = self.handle_tool_errors
            elif callable(self.handle_tool_errors):
                handled_types = _infer_handled_types(self.handle_tool_errors)
            else:
                # default behavior is catching all exceptions
                handled_types = (Exception,)

            # Unhandled
            if not self.handle_tool_errors or not isinstance(e, handled_types):
                raise e
            # Handled
            else:
                content = _handle_tool_error(e, flag=self.handle_tool_errors)

        return ToolMessage(
            content=content, name=call["name"], tool_call_id=call["id"], status="error"
        )

    async def _arun_one_before(self, call: ToolCall, config: RunnableConfig) -> ToolMessage:

        logger = logging.getLogger("HaleyAgentLogger")

        if invalid_tool_message := self._validate_tool_call(call):
            return invalid_tool_message
        try:
            input = {**call, **{"type": "tool_call"}}

            # input_updated = self.convert_input(input)

            logger.info(f"Async Tool Call Invoking: {self.tools_by_name[call['name']]}")

            args = input.get("args")

            # args_str = json.dumps(args)

            # tool_input = {"tool_input": args}

            # input["args"] = {"args": args}

            logger.info(f"Async Tool Call Invoking Input: {input}")

            tool_message: ToolMessage = await self.tools_by_name[call["name"]].ainvoke(
                input, config
            )

            input_tool_name = input.get("name", None)

            ############################################################
            # Temp work-around until tool calling schema issue resolved
            if input_tool_name == "CallAgentTool":
                input_args: dict = input.get("args", None)

                if input_args:
                    arg1_value = input_args.get("__arg1", None)

                    if arg1_value is not None and type(arg1_value) is str:
                        arg1_dict = json.loads(arg1_value)
                        input_args["__arg1"] = {"agent_request": arg1_dict}

                    if arg1_value is not None and type(arg1_value) is dict:
                        input_args["__arg1"] = {"agent_request": arg1_value}

            logger.info(f"Tool Call Invoking Updated Input: {input}")

            ############################################################

            tool_name = call['name']

            reasoning_message = {
                "agent_thought": f"calling tool: {tool_name}"
            }

            if self.reasoning_queue:
                await self.reasoning_queue.put(reasoning_message)

            tool_message: ToolMessage = self.tools_by_name[call["name"]].invoke(
                input, config
            )

            reasoning_message = {
                "agent_thought": f"completed calling tool: {tool_name}"
            }

            if self.reasoning_queue:
                await self.reasoning_queue.put(reasoning_message)

            # TODO: handle this properly in core
            tool_message.content = str_output(tool_message.content)
            return tool_message
        except Exception as e:
            if not self.handle_tool_errors:
                raise e
            content = TOOL_CALL_ERROR_TEMPLATE.format(error=repr(e))
            return ToolMessage(content, name=call["name"], tool_call_id=call["id"])

    def _parse_input(
            self,
            input: Union[
                list[AnyMessage],
                dict[str, Any],
                BaseModel,
            ],
            store: BaseStore,
    ) -> Tuple[List[ToolCall], Literal["list", "dict"]]:
        if isinstance(input, list):
            output_type = "list"
            message: AnyMessage = input[-1]
        elif isinstance(input, dict) and (messages := input.get(self.messages_key, [])):
            output_type = "dict"
            message = messages[-1]
        elif messages := getattr(input, self.messages_key, None):
            # Assume dataclass-like state that can coerce from dict
            output_type = "dict"
            message = messages[-1]
        else:
            raise ValueError("No message found in input")

        if not isinstance(message, AIMessage):
            raise ValueError("Last message is not an AIMessage")

        tool_calls = [
            self._inject_tool_args(call, input, store) for call in message.tool_calls
        ]
        return tool_calls, output_type

    def _validate_tool_call(self, call: ToolCall) -> Optional[ToolMessage]:
        if (requested_tool := call["name"]) not in self.tools_by_name:
            content = INVALID_TOOL_NAME_ERROR_TEMPLATE.format(
                requested_tool=requested_tool,
                available_tools=", ".join(self.tools_by_name.keys()),
            )
            return ToolMessage(
                content, name=requested_tool, tool_call_id=call["id"], status="error"
            )
        else:
            return None

    def _inject_state(
            self,
            tool_call: ToolCall,
            input: Union[
                list[AnyMessage],
                dict[str, Any],
                BaseModel,
            ],
    ) -> ToolCall:
        state_args = self.tool_to_state_args[tool_call["name"]]
        if state_args and isinstance(input, list):
            required_fields = list(state_args.values())
            if (
                    len(required_fields) == 1
                    and required_fields[0] == self.messages_key
                    or required_fields[0] is None
            ):
                input = {self.messages_key: input}
            else:
                err_msg = (
                    f"Invalid input to ToolNode. Tool {tool_call['name']} requires "
                    f"graph state dict as input."
                )
                if any(state_field for state_field in state_args.values()):
                    required_fields_str = ", ".join(f for f in required_fields if f)
                    err_msg += f" State should contain fields {required_fields_str}."
                raise ValueError(err_msg)
        if isinstance(input, dict):
            tool_state_args = {
                tool_arg: input[state_field] if state_field else input
                for tool_arg, state_field in state_args.items()
            }

        else:
            tool_state_args = {
                tool_arg: getattr(input, state_field) if state_field else input
                for tool_arg, state_field in state_args.items()
            }

        tool_call["args"] = {
            **tool_call["args"],
            **tool_state_args,
        }
        return tool_call

    def _inject_store(self, tool_call: ToolCall, store: BaseStore) -> ToolCall:
        store_arg = self.tool_to_store_arg[tool_call["name"]]
        if not store_arg:
            return tool_call

        if store is None:
            raise ValueError(
                "Cannot inject store into tools with InjectedStore annotations - "
                "please compile your graph with a store."
            )

        tool_call["args"] = {
            **tool_call["args"],
            store_arg: store,
        }
        return tool_call

    def _inject_tool_args(
            self,
            tool_call: ToolCall,
            input: Union[
                list[AnyMessage],
                dict[str, Any],
                BaseModel,
            ],
            store: BaseStore,
    ) -> ToolCall:
        if tool_call["name"] not in self.tools_by_name:
            return tool_call

        tool_call_copy: ToolCall = copy(tool_call)
        tool_call_with_state = self._inject_state(tool_call_copy, input)
        tool_call_with_store = self._inject_store(tool_call_with_state, store)
        return tool_call_with_store


def tools_condition(
    state: Union[list[AnyMessage], dict[str, Any], BaseModel],
    messages_key: str = "messages",
) -> Literal["tools", "__end__"]:

    if isinstance(state, list):
        ai_message = state[-1]
    elif isinstance(state, dict) and (messages := state.get(messages_key, [])):
        ai_message = messages[-1]
    elif messages := getattr(state, messages_key, []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "__end__"


class InjectedState(InjectedToolArg):
    """Annotation for a Tool arg that is meant to be populated with the graph state.

    Any Tool argument annotated with InjectedState will be hidden from a tool-calling
    model, so that the model doesn't attempt to generate the argument. If using
    ToolNode, the appropriate graph state field will be automatically injected into
    the model-generated tool args.

    Args:
        field: The key from state to insert. If None, the entire state is expected to
            be passed in.

    Example:
        ```python
        from typing import List
        from typing_extensions import Annotated, TypedDict

        from langchain_core.messages import BaseMessage, AIMessage
        from langchain_core.tools import tool

        from langgraph.prebuilt import InjectedState, ToolNode


        class AgentState(TypedDict):
            messages: List[BaseMessage]
            foo: str

        @tool
        def state_tool(x: int, state: Annotated[dict, InjectedState]) -> str:
            '''Do something with state.'''
            if len(state["messages"]) > 2:
                return state["foo"] + str(x)
            else:
                return "not enough messages"

        @tool
        def foo_tool(x: int, foo: Annotated[str, InjectedState("foo")]) -> str:
            '''Do something else with state.'''
            return foo + str(x + 1)

        node = ToolNode([state_tool, foo_tool])

        tool_call1 = {"name": "state_tool", "args": {"x": 1}, "id": "1", "type": "tool_call"}
        tool_call2 = {"name": "foo_tool", "args": {"x": 1}, "id": "2", "type": "tool_call"}
        state = {
            "messages": [AIMessage("", tool_calls=[tool_call1, tool_call2])],
            "foo": "bar",
        }
        node.invoke(state)
        ```

        ```pycon
        [
            ToolMessage(content='not enough messages', name='state_tool', tool_call_id='1'),
            ToolMessage(content='bar2', name='foo_tool', tool_call_id='2')
        ]
        ```
    """  # noqa: E501

    def __init__(self, field: Optional[str] = None) -> None:
        self.field = field



class InjectedStore(InjectedToolArg):
    """Annotation for a Tool arg that is meant to be populated with LangGraph store.

    Any Tool argument annotated with InjectedStore will be hidden from a tool-calling
    model, so that the model doesn't attempt to generate the argument. If using
    ToolNode, the appropriate store field will be automatically injected into
    the model-generated tool args. Note: if a graph is compiled with a store object,
    the store will be automatically propagated to the tools with InjectedStore args
    when using ToolNode.

    !!! Warning
        `InjectedStore` annotation requires `langchain-core >= 0.3.8`

    Example:
        ```python
        from typing import Any
        from typing_extensions import Annotated

        from langchain_core.messages import AIMessage
        from langchain_core.tools import tool

        from langgraph.store.memory import InMemoryStore
        from langgraph.prebuilt import InjectedStore, ToolNode

        store = InMemoryStore()
        store.put(("values",), "foo", {"bar": 2})

        @tool
        def store_tool(x: int, my_store: Annotated[Any, InjectedStore()]) -> str:
            '''Do something with store.'''
            stored_value = my_store.get(("values",), "foo").value["bar"]
            return stored_value + x

        node = ToolNode([store_tool])

        tool_call = {"name": "store_tool", "args": {"x": 1}, "id": "1", "type": "tool_call"}
        state = {
            "messages": [AIMessage("", tool_calls=[tool_call])],
        }

        node.invoke(state, store=store)
        ```

        ```pycon
        {
            "messages": [
                ToolMessage(content='3', name='store_tool', tool_call_id='1'),
            ]
        }
        ```
    """  # noqa: E501


def _is_injection(
    type_arg: Any, injection_type: Union[Type[InjectedState], Type[InjectedStore]]
) -> bool:
    if isinstance(type_arg, injection_type) or (
        isinstance(type_arg, type) and issubclass(type_arg, injection_type)
    ):
        return True
    origin_ = get_origin(type_arg)
    if origin_ is Union or origin_ is Annotated:
        return any(_is_injection(ta, injection_type) for ta in get_args(type_arg))
    return False


def _get_state_args(tool: BaseTool) -> Dict[str, Optional[str]]:
    full_schema = tool.get_input_schema()
    tool_args_to_state_fields: Dict = {}

    for name, type_ in get_all_basemodel_annotations(full_schema).items():
        injections = [
            type_arg
            for type_arg in get_args(type_)
            if _is_injection(type_arg, InjectedState)
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


def _get_store_arg(tool: BaseTool) -> Optional[str]:
    full_schema = tool.get_input_schema()
    for name, type_ in get_all_basemodel_annotations(full_schema).items():
        injections = [
            type_arg
            for type_arg in get_args(type_)
            if _is_injection(type_arg, InjectedStore)
        ]
        if len(injections) > 1:
            ValueError(
                "A tool argument should not be annotated with InjectedStore more than "
                f"once. Received arg {name} with annotations {injections}."
            )
        elif len(injections) == 1:
            return name
        else:
            pass

    return None