import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Type, TypeVar

from pydantic import BaseModel, ValidationError

from kgraphplanner.vital_agent_rest_resource_client.tools.tool_parameters import ToolParameters
from kgraphplanner.vital_agent_rest_resource_client.tools.tool_results import ToolResults

logger = logging.getLogger(__name__)

M = TypeVar("M", bound=BaseModel)


def build_model(model: Type[M], data: Dict[str, Any]) -> M:
    """Build ``model`` from a wire dict, keeping every field it declares.

    Handlers used to copy named fields one at a time, so a field the service
    added never reached the agent until the handler was edited too. Building
    from the whole dict means a model change is enough.

    An optional field whose value doesn't validate is dropped (logged by name
    only, never by value) rather than failing the whole result. A missing or
    invalid required field still raises.
    """
    try:
        return model.model_validate(data)
    except ValidationError as e:
        required = {name for name, f in model.model_fields.items() if f.is_required()}
        bad = {err["loc"][0] for err in e.errors() if err.get("loc")}
        if not bad or bad & required:
            raise
        logger.warning(f"{model.__name__}: dropped field(s) that did not validate: {sorted(bad)}")
        return model.model_validate({k: v for k, v in data.items() if k not in bad})


class ToolHandler(ABC):

    @abstractmethod
    def handle_response(self, tool_parameters: ToolParameters, response: dict) -> ToolResults:
        pass
