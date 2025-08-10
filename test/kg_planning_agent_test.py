import logging
from dotenv import load_dotenv
from datetime import datetime
from rich.console import Console
from langchain.callbacks.base import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from kgraphplanner.agent.kg_planning_agent import KGPlanningAgent
from kgraphplanner.checkpointer.memory_checkpointer import MemoryCheckpointer
from kgraphplanner.tool_manager.tool_manager import ToolManager
from kgraphplanner.tools.place_search.place_search_tool import PlaceSearchTool
from kgraphplanner.tools_internal.kgraph_query.search_contacts_tool import SearchContactsTool
from kgraphplanner.tools.send_message.send_message_tool import SendMessageTool
from kgraphplanner.tools.weather.weather_info_tool import WeatherInfoTool


def print_stream(stream, messages_out: list = None):

    if messages_out is None:
        messages_out = []

    for s in stream:
        message = s["messages"][-1]
        messages_out.append(message)
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


def get_timestamp() -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return timestamp


class LoggingHandler(BaseCallbackHandler):
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        self.logger.info(f"LLM Request: {prompts}")

    def on_llm_end(self, response, **kwargs):
        self.logger.info(f"LLM Response: {response.generations}")


def main():
    print("KG Planning Agent In-Memory Test")

    load_dotenv()

    rich = Console()

    # uses langgraph for a planning/execution graph using tools
    # use in-memory checkpointer
    # use langchain tool definitions
    # use openai models with structured output

    # convert incoming message history into graph
    # which may include tool calls and responses

    # tools include external resources like weather api
    # and kgservice functions for queries and updates

    # updated objects are put into response payload in a container
    # kgservice is treated as read-only

    # initially this is one-shot of changed objects returned
    # next iteration will use websocket to request object modifications
    # with response from websocket confirming (or reverting) object changes
    # this could be modeled with a human-in-the-loop case,
    # so it's really "system" in the loop with the system confirming a transaction
    # or not if the update fails or conflicts

    # input:
    # send john a txt message with the weather in philadelphia
    # john --> contact info
    # philadelphia --> lat, long
    # lat, long --> weather
    # weather report --> txt message to john

    logging_handler = LoggingHandler()

    model = ChatOpenAI(model="gpt-4o", callbacks=[logging_handler], temperature=0)

    tool_endpoint = "http://localhost:8008"

    tool_config = {
        "tool_endpoint": tool_endpoint
    }

    tool_manager = ToolManager(tool_config)

    place_search_config = {
        "tool_endpoint": tool_endpoint,
        "place_search_tool": {}
    }
    weather_config = {
        "tool_endpoint": tool_endpoint,
        "weather_tool": {}
    }
    search_contacts_config = {
        "tool_endpoint": tool_endpoint,
        "search_contacts_tool": {}
    }
    send_message_config = {
        "tool_endpoint": tool_endpoint,
        "send_message_tool": {}
    }

    place_search_tool = PlaceSearchTool(place_search_config, tool_manager=tool_manager)

    weather_tool = WeatherInfoTool(weather_config, tool_manager=tool_manager)

    search_contacts_tool = SearchContactsTool(search_contacts_config, tool_manager=tool_manager)

    send_message_tool = SendMessageTool(send_message_config, tool_manager=tool_manager)

    # getting tools to use in agent into a function list
    tool_function_list = []

    for t in tool_manager.get_tool_list():
        tool_function_list.append(t.get_tool_function())

    # Note: this isn't used, the ToolNode handles executing tools
    # tool_executor = ToolExecutor(tools=tool_list)
    # If tool_executor passed in, the tool list is just used
    # to instantiate ToolNode and executor is ignored
    # So all logging and routing should go through the ToolNode

    memory = MemoryCheckpointer()

    agent = KGPlanningAgent(model=model, checkpointer=memory, tools=tool_function_list)

    graph = agent.compile()

    image_bytes = graph.get_graph().draw_mermaid_png()

    with open('output_image.png', 'wb') as image_file:
        image_file.write(image_bytes)

    # os.system('open output_image.png')

    config = {"configurable": {"thread_id": "urn:thread_1"}}

    system_message_content = """
    You are an artificial intelligence assistant named Haley.
    You are a 30 year-old professional woman.
    Your telephone number is 555-555-1212.
    If you look up a place location, re-use that same location as needed.
    If you generated a weather report for a location recently, you can re-use that weather report if it's for the same location.
    Weather reports should include information about the current weather right now, today's weather, and the next few days if available.
    """

    system_message = SystemMessage(content=system_message_content)

    content = f"{get_timestamp()}: what is the weather in philly"

    message_input = [
        system_message,
        HumanMessage(content=content)
    ]

    inputs = {"messages": message_input}

    messages_out = []

    print_stream(graph.stream(inputs, config, stream_mode="values"), messages_out)

    for m in messages_out:
        t = type(m)
        print(f"History ({t}): {m}")

    # exit(0)

    content = "who are you and who am I?"

    message_input = [
        system_message,
        HumanMessage(content=f"{get_timestamp()}: Hello, my name is Marc."),
        AIMessage(content=f"{get_timestamp()}: Hello Marc, I am Haley, How can I help you?"),
        HumanMessage(content=f"{get_timestamp()}: {content}")
    ]

    inputs = {"messages": message_input}

    for m in messages_out:
        t = type(m)
        print(f"History ({t}): {m}")

    messages_out = []

    print_stream(graph.stream(inputs, config, stream_mode="values"), messages_out)

    for m in messages_out:
        t = type(m)
        print(f"History ({t}): {m}")

    config = {"configurable": {"thread_id": "urn:thread_2"}}

    content = f"""
    {get_timestamp()}: get a weather report for philly and send it to John using txt and tell me if it worked.
    If you re-used information from a previous tool request, tell me what information was re-used.
    """

    message_input = [
        system_message,
        HumanMessage(content=content)
    ]

    inputs = {"messages": message_input}

    for m in messages_out:
        t = type(m)
        print(f"History ({t}): {m}")

    messages_out = []

    print_stream(graph.stream(inputs, config, stream_mode="values"), messages_out)

    for m in messages_out:
        t = type(m)
        print(f"History ({t}): {m}")


if __name__ == "__main__":
    main()
