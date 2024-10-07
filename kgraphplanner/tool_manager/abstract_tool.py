from abc import ABC, abstractmethod
from typing import Callable
from kgraphplanner.tool_manager.tool_request import ToolRequest
from kgraphplanner.tool_manager.tool_response import ToolResponse

# circular dependency
# from kgraphplanner.tool_manager.tool_manager import ToolManager


class AbstractTool(ABC):
    def __init__(self, config: dict, tool_manager=None, name: str = None):
        self.config = config
        self.tool_manager = tool_manager
        if name:
            self.name = name
        else:
            self.name = self.__class__.__name__
        if self.tool_manager:
            from kgraphplanner.tool_manager.tool_manager import ToolManager
            tm: ToolManager = tool_manager
            tm.add_tool(self)

    @classmethod
    def get_tool_cls_name(cls) -> str:
        return cls.__name__

    def get_tool_name(self) -> str:
        return self.name

    @abstractmethod
    def handle_request(self, request: ToolRequest) -> ToolResponse:
        pass

    @abstractmethod
    def get_tool_function(self) -> Callable:
        pass

    def get_generic_text(self) -> str:
        generic_text = """Across the street, the lights flicker occasionally as evening sets in.
    A cat meanders along the sidewalk, its attention briefly caught by a drifting leaf.
    From somewhere in the distance, the hum of traffic blends seamlessly with the quiet of the neighborhood.
    Trees line the road, their branches swaying gently in the breeze.
    This ordinary scene unfolds without hinting at anything out of the ordinary or urgent.
    """
        return generic_text

    """The time of day when people sit on park benches, observing birds, often passes quietly. 
    Sometimes, different colored cars will drive by, each heading in its own direction. 
    In the background, the sound of distant laughter or the occasional bark of a dog can be heard. 
    The sky above may shift from clear to cloudy, without any noticeable change in the atmosphere. 
    Throughout, a sense of normalcy prevails, with no specific events to alter the pace of daily life.
    """

    @abstractmethod
    def get_sample_text(self) -> str:
        pass

