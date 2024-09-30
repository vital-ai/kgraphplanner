from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
)

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.pydantic_v1 import BaseModel, ValidationError
from langchain_core.runnables import (
    RunnableConfig,
)
from langchain_core.runnables.config import get_executor_for_config
from langchain_core.tools import BaseTool, create_schema_from_function
from langgraph.utils.runnable import RunnableCallable
from pydantic import BaseModel as BaseModelV2
from pydantic import ValidationError as ValidationErrorV2



def _default_format_error(
    error: BaseException, call: ToolCall, schema: Type[BaseModel]
) -> str:
    """Default error formatting function."""
    return f"{repr(error)}\n\nRespond after fixing all validation errors."


class ValidationNode(RunnableCallable):

    def __init__(
        self,
        schemas: Sequence[Union[BaseTool, Type[BaseModel], Callable]],
        *,
        format_error: Optional[
            Callable[[BaseException, ToolCall, Type[BaseModel]], str]
        ] = None,
        name: str = "validation",
        tags: Optional[list[str]] = None,
    ) -> None:
        super().__init__(self._func, None, name=name, tags=tags, trace=False)
        self._format_error = format_error or _default_format_error
        self.schemas_by_name: Dict[str, Type[BaseModel]] = {}
        for schema in schemas:
            if isinstance(schema, BaseTool):
                if schema.args_schema is None:
                    raise ValueError(
                        f"Tool {schema.name} does not have an args_schema defined."
                    )
                self.schemas_by_name[schema.name] = schema.args_schema
            elif isinstance(schema, type) and issubclass(
                schema, (BaseModel, BaseModelV2)
            ):
                self.schemas_by_name[schema.__name__] = cast(Type[BaseModel], schema)
            elif callable(schema):
                base_model = create_schema_from_function("Validation", schema)
                self.schemas_by_name[schema.__name__] = base_model
            else:
                raise ValueError(
                    f"Unsupported input to ValidationNode. Expected BaseModel, tool or function. Got: {type(schema)}."
                )

    def _get_message(
        self, input: Union[list[AnyMessage], dict[str, Any]]
    ) -> Tuple[str, AIMessage]:
        """Extract the last AIMessage from the input."""
        if isinstance(input, list):
            output_type = "list"
            messages: list = input
        elif messages := input.get("messages", []):
            output_type = "dict"
        else:
            raise ValueError("No message found in input")
        message: AnyMessage = messages[-1]
        if not isinstance(message, AIMessage):
            raise ValueError("Last message is not an AIMessage")
        return output_type, message

    def _func(
        self, input: Union[list[AnyMessage], dict[str, Any]], config: RunnableConfig
    ) -> Any:
        """Validate and run tool calls synchronously."""
        output_type, message = self._get_message(input)

        def run_one(call: ToolCall):
            schema = self.schemas_by_name[call["name"]]
            try:
                output = schema.validate(call["args"])
                return ToolMessage(
                    content=output.json(),
                    name=call["name"],
                    tool_call_id=cast(str, call["id"]),
                )
            except (ValidationError, ValidationErrorV2) as e:
                return ToolMessage(
                    content=self._format_error(e, call, schema),
                    name=call["name"],
                    tool_call_id=cast(str, call["id"]),
                    additional_kwargs={"is_error": True},
                )

        with get_executor_for_config(config) as executor:
            outputs = [*executor.map(run_one, message.tool_calls)]
            if output_type == "list":
                return outputs
            else:
                return {"messages": outputs}

