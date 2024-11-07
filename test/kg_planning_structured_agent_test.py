import logging
from typing import List, Any
from dotenv import load_dotenv
from datetime import datetime
from rich.console import Console
from langchain.callbacks.base import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from typing_extensions import TypedDict
from kgraphplanner.agent.kg_planning_structured_agent import KGPlanningStructuredAgent
from kgraphplanner.checkpointer.memory_checkpointer import MemoryCheckpointer
from kgraphplanner.tool_manager.tool_manager import ToolManager
from kgraphplanner.tools.place_search.place_search_tool import PlaceSearchTool
from kgraphplanner.tools.weather.current_weather_tool import CurrentWeatherTool
from kgraphplanner.tools_internal.kgraph_query.search_contacts_tool import SearchContactsTool
from kgraphplanner.tools.send_message.send_message_tool import SendMessageTool
from kgraphplanner.tools.weather.weather_info_tool import WeatherInfoTool
import pprint
import threading
import queue

def print_stream(stream, messages_out: list = []) -> TypedDict:

    pp = pprint.PrettyPrinter(indent=4, width=40)

    final_result = None

    iteration = 0

    for s in stream:

        print(s)

        iteration += 1

        messages = s["messages"]

        for m in messages:
            t = type(m)
            print(f"Stream {iteration}: History ({t}): {m}")

        final_response = s.get("final_response", None)

        if final_response:
            print(s)
            final_result = s
        else:

            # parallel tools add more than one message,
            # but this is just adding the last one so it's missing some
            message = s["messages"][-1]
            messages_out.append(message)
            if isinstance(message, tuple):
                print(message)
            else:
                message.pretty_print()
            # final_result = s

    print(final_result)

    response = final_result.get("final_response", None)

    print("Final_Result:\n")
    print("--------------------------------------")
    pp.pprint(response)
    print("--------------------------------------")
    return response


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
        self.logger.info(f"LLM Request: {serialized}")
        self.logger.info(f"LLM Request Prompts: {prompts}")
        self.logger.info(f"LLM Request KWArgs: {kwargs}")


    def on_llm_end(self, response, **kwargs):
        self.logger.info(f"LLM Response: {response}")
        self.logger.info(f"LLM Response Generations: {response.generations}")
        self.logger.info(f"LLM Response KWArgs: {kwargs}")


system_message_content = """
    You are an artificial intelligence assistant named Haley.
    You are a 30 year-old professional woman.
    Your telephone number is 555-555-1212.
    If you look up a place location, re-use that same location as needed.
    """


def case_one(tool_manager, graph):

    pp = pprint.PrettyPrinter(indent=4, width=40)

    config = {"configurable": {"thread_id": "urn:thread_1"}}

    system_message = SystemMessage(content=system_message_content)

    # content = f"{get_timestamp()}: hello there!"

    content = f"{get_timestamp()}: what is the weather in LA and Philly?"

    # content = f"{get_timestamp()}: what was the weather in LA and Philly on Christmas in 2020?"

    # content = f"{get_timestamp()}: what is the weather in Philly?"


    # handle prior messages
    # TODO add in incoming history
    # chat_message_list = [
    #    ("system", system_prompt)
    # ]

    # for h in history_list:
    #    chat_message_list.append(h)

    # instead of adding them directly, create a single merged HumanMessage
    # to act as "summary" of prior activity, which would include tool requests/responses
    # prior tool messages may need the guids to refer to and retrieve from the cache
    # or retrieve from kg, but hopefully what is in the messages is sufficient.
    # before we put json of tool replies into history, but that can slow down structured response calls
    # so potentially convert tool responses to markdown or similar first

    # we'll be returning just the new messages to add to an interaction, so we'll skip this
    # summary message in our reply

    message_input = [
        system_message,
        HumanMessage(content=content)
    ]

    inputs = {"messages": message_input}

    messages_out = []

    agent_status_response = print_stream(graph.stream(inputs, config, stream_mode="values"), messages_out)

    for m in messages_out:
        t = type(m)
        print(f"History ({t}): {m}")

    human_text_request = agent_status_response.get("human_text_request", None)
    agent_text_response = agent_status_response.get("agent_text_response", None)
    agent_request_status = agent_status_response.get("agent_request_status", None)
    agent_payload_list = agent_status_response.get("agent_payload_list", [])
    agent_request_status_message = agent_status_response.get("agent_request_status_message", None)
    missing_input = agent_status_response.get("missing_input", None)

    print(f"Status: {agent_request_status}")

    print(f"Human Text: {human_text_request}")
    print(f"Agent Text: {agent_text_response}")

    for agent_payload in agent_payload_list:

        pp.pprint(agent_payload)

        payload_class_name = agent_payload.get("payload_class_name", None)
        payload_guid = agent_payload.get("payload_guid", None)

        print("--------------------------------------")
        print(f"Agent GUID: {payload_guid}")
        print(f"Agent Class Name: {payload_class_name}")
        print("--------------------------------------")


def case_two(tool_manager, graph):

    config = {"configurable": {"thread_id": "urn:thread_1"}}

    system_message = SystemMessage(content=system_message_content)

    content = "who are you and who am I?"

    message_input = [
        system_message,
        HumanMessage(content=f"{get_timestamp()}: Hello, my name is Marc."),
        AIMessage(content=f"{get_timestamp()}: Hello Marc, I am Haley, How can I help you?"),
        HumanMessage(content=f"{get_timestamp()}: {content}")
    ]

    inputs = {"messages": message_input}

    messages_out = []

    print_stream(graph.stream(inputs, config, stream_mode="values"), messages_out)

    for m in messages_out:
        t = type(m)
        print(f"History ({t}): {m}")

def case_three(tool_manager, graph):

    config = {"configurable": {"thread_id": "urn:thread_2"}}

    system_message = SystemMessage(content=system_message_content)

    content = f"""
        {get_timestamp()}: get a weather report for philly and send it to John using txt and tell me if it worked.
        If you re-used information from a previous tool request, tell me what information was re-used.
        """

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


def reasoning_consumer(message_queue: queue.Queue, stop_event: threading.Event):

    while not stop_event.is_set():
        try:
            message = message_queue.get(timeout=0.5)
            print(f"Consumed: {message}")
            if message == "STOP":
                break

            # send back reasoning on websocket in AIMP format
            # for display in UI

            # messages to include some degree of context of activity
            # as well as errors, re-starts, changing of mind, etc.

            message_queue.task_done()
        except queue.Empty:
            continue


def main():
    print("KG Planning Structured Agent Test")

    load_dotenv()

    rich = Console()

    logging_handler = LoggingHandler()

    message_queue = queue.Queue()

    stop_event = threading.Event()

    consumer_thread = threading.Thread(
        target=reasoning_consumer,
        args=(message_queue, stop_event),
        daemon=True)

    consumer_thread.start()

    model_tools = ChatOpenAI(model="gpt-4o", callbacks=[logging_handler], temperature=0)
    model_structured = ChatOpenAI(model="gpt-4o", callbacks=[logging_handler], temperature=0)

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
    current_weather_config = {
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

    weather_tool = WeatherInfoTool(weather_config,  tool_manager=tool_manager)

    current_weather_tool = CurrentWeatherTool(current_weather_config,  tool_manager=tool_manager)

    search_contacts_tool = SearchContactsTool(search_contacts_config,  tool_manager=tool_manager)

    send_message_tool = SendMessageTool(send_message_config,  tool_manager=tool_manager)

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

    agent = KGPlanningStructuredAgent(
        model_tools=model_tools,
        model_structured=model_structured,
        checkpointer=memory,
        tools=tool_function_list,
        reasoning_queue=message_queue
    )

    graph = agent.compile()

    image_bytes = graph.get_graph().draw_mermaid_png()

    with open('output_image.png', 'wb') as image_file:
        image_file.write(image_bytes)

    # os.system('open output_image.png')

    case_one(tool_manager, graph)

    # case_two(tool_manager, graph)

    # case_three(tool_manager, graph)

    message_queue.put("STOP")

    stop_event.set()
    consumer_thread.join()


if __name__ == "__main__":
    main()
