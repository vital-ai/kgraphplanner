import logging
from typing import List, Any
from dotenv import load_dotenv
from datetime import datetime
from rich.console import Console
from langchain.callbacks.base import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from typing_extensions import TypedDict
from kgraphplanner.agent.kg_planning_agent import KGPlanningStructuredAgent
from kgraphplanner.checkpointer.memory_checkpointer import MemoryCheckpointer
from kgraphplanner.tool_manager.tool_manager import ToolManager
from kgraphplanner.tools.place_search.place_search_tool import PlaceSearchTool
from kgraphplanner.tools_internal.kgraph_query.search_contacts_tool import SearchContactsTool
from kgraphplanner.tools.send_message.send_message_tool import SendMessageTool
from kgraphplanner.tools.weather.weather_info_tool import WeatherInfoTool
import pprint


def print_stream(stream, messages_out: list = []) -> TypedDict:

    pp = pprint.PrettyPrinter(indent=4, width=40)

    final_result = None

    for s in stream:
        message = s["messages"][-1]
        messages_out.append(message)
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()
        final_result = s

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
        self.logger.info(f"LLM Request: {prompts}")

    def on_llm_end(self, response, **kwargs):
        self.logger.info(f"LLM Response: {response.generations}")


system_message_content = """
    You are an artificial intelligence assistant named Haley.
    You are a 30 year-old professional woman.
    Your telephone number is 555-555-1212.
    If you look up a place location, re-use that same location as needed.
    """


# the guids should be listed in the structure response output
def extract_tool_response_data(tool_manager, messages_out: List[Any]) -> List[TypedDict]:
    capture_guids = []
    capture_classes = {}
    for message in messages_out:
        if hasattr(message, 'tool_calls'):
            for call in message.tool_calls:
                if call['name'] == 'capture_response':
                    args = call.get('args', {})
                    guid = args.get('tool_response_guid')
                    class_name = args.get('response_class_name')
                    if guid and class_name:
                        capture_guids.append(guid)
                        capture_classes[guid] = class_name

    print(f"capture_guids: {capture_guids}")
    print(f"capture_classes: {capture_classes}")

    captured_responses = []

    for guid in capture_guids:
        tool_data = tool_manager.get_tool_cache().get_from_cache(guid)
        if tool_data:
            captured_responses.append(tool_data)

    return captured_responses


def case_one(tool_manager, graph):

    pp = pprint.PrettyPrinter(indent=4, width=40)

    config = {"configurable": {"thread_id": "urn:thread_1"}}

    system_message = SystemMessage(content=system_message_content)

    content = f"{get_timestamp()}: what was the weather in LA and Philly on Christmas in 2020?"

    # content = f"{get_timestamp()}: what is the weather in LA and Philly?"

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

    # there may be minor variants for cases of capturing knowledge graph objects
    # objects for the UI, etc.

    tool_response_data_list = extract_tool_response_data(tool_manager, messages_out)

    # print(tool_response_data_list)

    for tool_response_data in tool_response_data_list:
        print(f"Response Data:\n{tool_response_data}\n")

    human_text_request = agent_status_response.get("human_text_request", None)
    agent_text_response = agent_status_response.get("agent_text_response", None)
    agent_request_status = agent_status_response.get("agent_request_status", None)
    agent_include_payload = agent_status_response.get("agent_include_payload", None)
    agent_payload_class_list = agent_status_response.get("agent_payload_class_list", None)
    agent_payload_guid_list = agent_status_response.get("agent_payload_guid_list", None)
    agent_request_status_message = agent_status_response.get("agent_request_status_message", None)
    missing_input = agent_status_response.get("missing_input", None)

    print(f"Status: {agent_request_status}")

    print(f"Human Text: {human_text_request}")
    print(f"Agent Text: {agent_text_response}")

    if agent_include_payload:
        print(f"Agent Payload ClassList: {agent_payload_class_list}")
        print(f"Agent Payload GuidList: {agent_payload_guid_list}")

        for guid in agent_payload_guid_list:

            response_obj = tool_manager.get_tool_cache().get_from_cache(guid)
            tool_data_class = response_obj.get("tool_data_class", None)

            print("--------------------------------------")
            print(f"Tool Data Class: {tool_data_class}")
            pp.pprint(response_obj)
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


def main():
    print("KG Planning Structured Agent Test")

    load_dotenv()

    rich = Console()

    logging_handler = LoggingHandler()

    model_tools = ChatOpenAI(model="gpt-4o", callbacks=[logging_handler], temperature=0)
    model_structured = ChatOpenAI(model="gpt-4o", callbacks=[logging_handler], temperature=0)

    tool_config = {}
    tool_manager = ToolManager(tool_config)

    place_search_tool = PlaceSearchTool({}, tool_manager=tool_manager)
    weather_tool = WeatherInfoTool({},  tool_manager=tool_manager)
    search_contacts_tool = SearchContactsTool({},  tool_manager=tool_manager)
    send_message_tool = SendMessageTool({},  tool_manager=tool_manager)

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

    agent = KGPlanningStructuredAgent(model_tools=model_tools, model_structured=model_structured, checkpointer=memory, tools=tool_function_list)

    graph = agent.compile()

    image_bytes = graph.get_graph().draw_mermaid_png()

    with open('output_image.png', 'wb') as image_file:
        image_file.write(image_bytes)

    # os.system('open output_image.png')

    case_one(tool_manager, graph)

    # case_two(tool_manager, graph)

    # case_three(tool_manager, graph)


if __name__ == "__main__":
    main()
