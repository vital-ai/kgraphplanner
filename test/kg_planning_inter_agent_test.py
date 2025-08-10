import logging
from typing import Sequence
from dotenv import load_dotenv
from datetime import datetime
from langchain_core.tools import Tool, BaseTool
from rich.console import Console
from langchain.callbacks.base import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from typing import TypedDict
from kgraphplanner.agent.kg_planning_inter_agent import KGPlanningInterAgent
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


def process_stream(stream, messages_out: list) -> TypedDict:

    logger = logging.getLogger("HaleyAgentLogger")

    pp = pprint.PrettyPrinter(indent=4, width=40)

    final_result = None

    iteration = 0

    for s in stream:

        logger.info(s)

        iteration += 1

        messages = s["messages"]

        for m in messages:
            t = type(m)
            logger.info(f"Stream {iteration}: History ({t}): {m}")

        final_response = s.get("final_response", None)

        if final_response:
            logger.info(s)
            final_result = s
        else:
            # parallel tools add more than one message,
            # but this is just adding the last one so it's missing some
            message = s["messages"][-1]
            messages_out.append(message)
            if isinstance(message, tuple):
                logger.info(message)
            else:
                message.pretty_print()
            # final_result = s

    logger.info(final_result)

    response = final_result.get("final_response", None)

    logger.info("Final_Result:\n")
    logger.info("--------------------------------------")
    logger.info(pp.pformat(response))
    logger.info("--------------------------------------")
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
    
    You will provide assistance to accomplish goals asked of you.
    
    You can use tools to complete your goals, such as answering questions or providing services.
    
    You can ask other assistants ("agents") to help with your goals.

    You must always try to use tools first and only ask other agents when you don't have a necessary tool to complete your goals yourself.
    
    You are charged $20 every time you ask another agent, and tool calls are free.  You want to minimize your costs.
    
    If it is necessary to ask another agent for help, you must register that request using the Call Agent tool,
    and then you will be called back once the response arrives.
    
    You should review your message history to determine if a previous tool request or response from an agent is available that helps complete your goals.
    
    If you must ask another assistant for help, you must make any other necessary agent calls at the same time
    so that the other assistants can work on them in parallel and you will get responses sooner and complete your goals more quickly.
    
    Begin List of Available Tools:
    --------------------
  
    
    Place_Search
    Search_Contacts
    Send_Message
    CallAgentTool
    --------------------
    End Available Tools
    
    Begin List of Available Agents:
    --------------------
    Name: AgentWeather
    Request Schema:
    
        class AgentWeatherRequest(BaseAgentRequest):
        
            Use this to get weather information for today and the next few days given the name of a place.
        
            Args:
                place_name_list (List[str]): The place name of the weather location, you may list multiple place names in a comma separated list

    Response Schema:
    
        class AgentWeatherResponse(BaseAgentResponse):
        
            Represents a daily weather prediction.
        
            Attributes:
                place_name (str): The place name of the weather location
                weatherReportText (str): Text giving the weather report.
                weatherReport (WeatherData): Structured data of the weather report.

    --------------------
    End Available Agents.
    
    --------------------
    BaseAgentRequest schema:
    
        class BaseAgentRequest(BaseModel):
        
            Represents the base class to collect the parameters for a request to an Agent
        
        Args:
            request_class_name (str): The name of the class of the request, which must be a subclass of BaseAgentRequest.
            agent_name (str): The name of the agent being called.
            agent_call_justification (str): A justification for why the agent being called to complete a goal.

    CallAgentCapture schema:
    
        class CallAgentCapture(BaseModel)
            Arguments for capturing a request to an Agent.
        
        Args:
            agent_request: Type[BaseAgentRequest]: The details of the Agent request which must be in an instance of a subclass of BaseAgentRequest


    --------------------
    """

# Current_Weather_Tool
# Weather_Tool

# ----------------------
# Previous Agent Responses:
# ---------------------
# History (<class 'langchain_core.messages.tool.ToolMessage'>): content='{"tool_request_guid": "a443860d-389c-43d7-8847-9483188f72a0", "tool_data_class": "WeatherData", "tool_name": "weather_tool", "tool_parameters": {"place_label": "Los Angeles", "latitude": 34.0549076, "longitude": -118.242643, "include_previous": false, "use_archive": false, "archive_date": ""}, "place_label": "Los Angeles", "latitude": 34.060257, "longitude": -118.23433, "timezone": "America/New_York", "weather_code": 0, "weather_code_description": "Sunny", "temperature": 74.9, "humidity": 29, "wind_speed": 7.9, "apparent_temperature": 69.3, "is_day": 1, "precipitation": 0.0, "precipitation_probability": 0, "cloud_cover": 0, "wind_direction_10m": 255, "wind_gusts_10m": 9.4, "daily_predictions": [{"date": "2024-11-10", "weather_code": 3, "weather_code_description": "Cloudy", "temperature_max": 78.6, "temperature_min": 49.2, "apparent_temperature_max": 73.8, "apparent_temperature_min": 42.6, "sunrise": "2024-11-10T09:21", "sunset": "2024-11-10T19:52", "precipitation_sum": 0.0, "precipitation_hours": 0.0, "precipitation_probability_max": 0, "precipitation_probability_min": 0, "precipitation_probability_mean": 0, "daylight_duration": 37845.71, "uv_index_max": 4.6, "wind_gusts_10m_max": 9.6}, {"date": "2024-11-11", "weather_code": 3, "weather_code_description": "Cloudy", "temperature_max": 70.6, "temperature_min": 47.0, "apparent_temperature_max": 66.6, "apparent_temperature_min": 40.3, "sunrise": "2024-11-11T09:22", "sunset": "2024-11-11T19:51", "precipitation_sum": 0.0, "precipitation_hours": 0.0, "precipitation_probability_max": 0, "precipitation_probability_min": 0, "precipitation_probability_mean": 0, "daylight_duration": 37747.64, "uv_index_max": 4.6, "wind_gusts_10m_max": 11.4}, {"date": "2024-11-12", "weather_code": 3, "weather_code_description": "Cloudy", "temperature_max": 69.2, "temperature_min": 48.8, "apparent_temperature_max": 64.2, "apparent_temperature_min": 45.5, "sunrise": "2024-11-12T09:23", "sunset": "2024-11-12T19:50", "precipitation_sum": 0.0, "precipitation_hours": 0.0, "precipitation_probability_max": 20, "precipitation_probability_min": 0, "precipitation_probability_mean": 3, "daylight_duration": 37650.6, "uv_index_max": 4.4, "wind_gusts_10m_max": 10.1}, {"date": "2024-11-13", "weather_code": 0, "weather_code_description": "Sunny", "temperature_max": 73.7, "temperature_min": 60.6, "apparent_temperature_max": 66.9, "apparent_temperature_min": 54.4, "sunrise": "2024-11-13T09:24", "sunset": "2024-11-13T19:50", "precipitation_sum": 0.0, "precipitation_hours": 0.0, "precipitation_probability_max": 1, "precipitation_probability_min": 0, "precipitation_probability_mean": 0, "daylight_duration": 37554.7, "uv_index_max": 4.55, "wind_gusts_10m_max": 9.4}, {"date": "2024-11-14", "weather_code": 2, "weather_code_description": "Partly cloudy", "temperature_max": 72.3, "temperature_min": 60.7, "apparent_temperature_max": 66.0, "apparent_temperature_min": 56.1, "sunrise": "2024-11-14T09:25", "sunset": "2024-11-14T19:49", "precipitation_sum": 0.0, "precipitation_hours": 0.0, "precipitation_probability_max": 1, "precipitation_probability_min": 0, "precipitation_probability_mean": 0, "daylight_duration": 37460.05, "uv_index_max": 4.5, "wind_gusts_10m_max": 10.1}, {"date": "2024-11-15", "weather_code": 1, "weather_code_description": "Sunny and Overcast", "temperature_max": 64.6, "temperature_min": 57.0, "apparent_temperature_max": 59.8, "apparent_temperature_min": 54.3, "sunrise": "2024-11-15T09:26", "sunset": "2024-11-15T19:49", "precipitation_sum": 0.0, "precipitation_hours": 0.0, "precipitation_probability_max": 15, "precipitation_probability_min": 1, "precipitation_probability_mean": 9, "daylight_duration": 37366.78, "uv_index_max": 4.2, "wind_gusts_10m_max": 13.0}, {"date": "2024-11-16", "weather_code": 51, "weather_code_description": "Light Rain", "temperature_max": 65.1, "temperature_min": 55.5, "apparent_temperature_max": 57.7, "apparent_temperature_min": 53.0, "sunrise": "2024-11-16T09:27", "sunset": "2024-11-16T19:48", "precipitation_sum": 0.012, "precipitation_hours": 3.0, "precipitation_probability_max": 24, "precipitation_probability_min": 15, "precipitation_probability_mean": 21, "daylight_duration": 37275.0, "uv_index_max": 2.5, "wind_gusts_10m_max": 13.2}]}' name='CurrentWeatherTool' id='507b2e93-8900-4482-b67c-eb2a596cb06b' tool_call_id='call_saKvnn2hyvBH9RcgQS6QkWQO'
# History (<class 'langchain_core.messages.tool.ToolMessage'>): content='{"tool_request_guid": "8b5dbcb1-1e6b-4f15-8a74-33f4462f9157", "tool_data_class": "WeatherData", "tool_name": "weather_tool", "tool_parameters": {"place_label": "Philadelphia", "latitude": 39.9525839, "longitude": -75.1652215, "include_previous": false, "use_archive": false, "archive_date": ""}, "place_label": "Philadelphia", "latitude": 39.96187, "longitude": -75.15539, "timezone": "America/New_York", "weather_code": 3, "weather_code_description": "Cloudy", "temperature": 51.6, "humidity": 68, "wind_speed": 3.3, "apparent_temperature": 48.1, "is_day": 0, "precipitation": 0.0, "precipitation_probability": 2, "cloud_cover": 100, "wind_direction_10m": 164, "wind_gusts_10m": 10.7, "daily_predictions": [{"date": "2024-11-10", "weather_code": 53, "weather_code_description": "Moderate Rain", "temperature_max": 55.3, "temperature_min": 34.8, "apparent_temperature_max": 51.3, "apparent_temperature_min": 29.5, "sunrise": "2024-11-10T06:40", "sunset": "2024-11-10T16:48", "precipitation_sum": 0.039, "precipitation_hours": 2.0, "precipitation_probability_max": 75, "precipitation_probability_min": 0, "precipitation_probability_mean": 10, "daylight_duration": 36457.47, "uv_index_max": 2.85, "wind_gusts_10m_max": 29.5}, {"date": "2024-11-11", "weather_code": 63, "weather_code_description": "Moderate Rain", "temperature_max": 64.8, "temperature_min": 49.4, "apparent_temperature_max": 63.2, "apparent_temperature_min": 45.4, "sunrise": "2024-11-11T06:41", "sunset": "2024-11-11T16:47", "precipitation_sum": 0.413, "precipitation_hours": 6.0, "precipitation_probability_max": 79, "precipitation_probability_min": 0, "precipitation_probability_mean": 18, "daylight_duration": 36334.38, "uv_index_max": 3.35, "wind_gusts_10m_max": 32.0}, {"date": "2024-11-12", "weather_code": 1, "weather_code_description": "Sunny and Overcast", "temperature_max": 53.4, "temperature_min": 41.6, "apparent_temperature_max": 43.9, "apparent_temperature_min": 33.1, "sunrise": "2024-11-12T06:43", "sunset": "2024-11-12T16:46", "precipitation_sum": 0.0, "precipitation_hours": 0.0, "precipitation_probability_max": 1, "precipitation_probability_min": 0, "precipitation_probability_mean": 0, "daylight_duration": 36212.54, "uv_index_max": 3.25, "wind_gusts_10m_max": 30.4}, {"date": "2024-11-13", "weather_code": 3, "weather_code_description": "Cloudy", "temperature_max": 51.5, "temperature_min": 39.9, "apparent_temperature_max": 43.6, "apparent_temperature_min": 32.1, "sunrise": "2024-11-13T06:44", "sunset": "2024-11-13T16:45", "precipitation_sum": 0.0, "precipitation_hours": 0.0, "precipitation_probability_max": 1, "precipitation_probability_min": 0, "precipitation_probability_mean": 1, "daylight_duration": 36092.05, "uv_index_max": 3.3, "wind_gusts_10m_max": 25.1}, {"date": "2024-11-14", "weather_code": 81, "weather_code_description": "Moderate Rain Showers", "temperature_max": 53.0, "temperature_min": 44.8, "apparent_temperature_max": 46.5, "apparent_temperature_min": 38.9, "sunrise": "2024-11-14T06:45", "sunset": "2024-11-14T16:44", "precipitation_sum": 0.677, "precipitation_hours": 10.0, "precipitation_probability_max": 19, "precipitation_probability_min": 1, "precipitation_probability_mean": 12, "daylight_duration": 35973.11, "uv_index_max": 0.9, "wind_gusts_10m_max": 19.0}, {"date": "2024-11-15", "weather_code": 53, "weather_code_description": "Moderate Rain", "temperature_max": 54.7, "temperature_min": 48.6, "apparent_temperature_max": 50.2, "apparent_temperature_min": 44.5, "sunrise": "2024-11-15T06:46", "sunset": "2024-11-15T16:44", "precipitation_sum": 0.051, "precipitation_hours": 5.0, "precipitation_probability_max": 16, "precipitation_probability_min": 5, "precipitation_probability_mean": 8, "daylight_duration": 35855.83, "uv_index_max": 0.4, "wind_gusts_10m_max": 25.5}, {"date": "2024-11-16", "weather_code": 2, "weather_code_description": "Partly cloudy", "temperature_max": 58.0, "temperature_min": 47.0, "apparent_temperature_max": 49.5, "apparent_temperature_min": 40.2, "sunrise": "2024-11-16T06:47", "sunset": "2024-11-16T16:43", "precipitation_sum": 0.0, "precipitation_hours": 0.0, "precipitation_probability_max": 5, "precipitation_probability_min": 1, "precipitation_probability_mean": 2, "daylight_duration": 35740.39, "uv_index_max": 2.95, "wind_gusts_10m_max": 25.5}]}' name='CurrentWeatherTool' id='d9003c3e-1fc6-4c1e-8d70-3018731e75a7' tool_call_id='call_D02OjxG6XN6aZ43qXHBThiDC'
#


def case_one(tool_manager, graph):

    logger = logging.getLogger("HaleyAgentLogger")

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

    agent_status_response = process_stream(graph.stream(inputs, config, stream_mode="values"), messages_out)

    for m in messages_out:
        t = type(m)
        logger.info(f"History ({t}): {m}")

    response_class_name = agent_status_response.get("response_class_name", None)

    if response_class_name == "AgentPayloadResponse":
        human_text_request = agent_status_response.get("human_text_request", None)
        agent_text_response = agent_status_response.get("agent_text_response", None)
        agent_request_status = agent_status_response.get("agent_request_status", None)
        agent_payload_list = agent_status_response.get("agent_payload_list", [])
        agent_request_status_message = agent_status_response.get("agent_request_status_message", None)
        missing_input = agent_status_response.get("missing_input", None)

        logger.info(f"Status: {agent_request_status}")

        logger.info(f"Human Text: {human_text_request}")
        logger.info(f"Agent Text: {agent_text_response}")

        for agent_payload in agent_payload_list:

            logger.info(pp.pformat(agent_payload))

            payload_class_name = agent_payload.get("payload_class_name", None)
            payload_guid = agent_payload.get("payload_guid", None)

            logger.info("--------------------------------------")
            logger.info(f"Agent GUID: {payload_guid}")
            logger.info(f"Agent Class Name: {payload_class_name}")
            logger.info("--------------------------------------")

    if response_class_name == "AgentCallResponse":
        logger.info("--------------------------------------")
        logger.info("Agent Call Response List:")
        agent_call_list = agent_status_response.get("agent_call_list", [])
        for call_capture in agent_call_list:
            agent_call_guid = call_capture.get("agent_call_guid", None)
            agent_call_request = call_capture.get("agent_call_request", None)

            logger.info(pp.pformat(call_capture))
        print("--------------------------------------")

def case_two(tool_manager, graph):

    logger = logging.getLogger("HaleyAgentLogger")

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

    process_stream(graph.stream(inputs, config, stream_mode="values"), messages_out)

    for m in messages_out:
        t = type(m)
        logger.info(f"History ({t}): {m}")


def case_three(tool_manager, graph):

    logger = logging.getLogger("HaleyAgentLogger")

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

    process_stream(graph.stream(inputs, config, stream_mode="values"), messages_out)

    for m in messages_out:
        t = type(m)
        logger.info(f"History ({t}): {m}")


def reasoning_consumer(message_queue: queue.Queue, stop_event: threading.Event):

    logger = logging.getLogger("HaleyAgentLogger")

    while not stop_event.is_set():
        try:
            message = message_queue.get(timeout=0.5)
            logger.info(f"Consumed: {message}")
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
    print("KG Planning Inter Agent Test")

    logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger("HaleyAgentLogger")

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

    model_tools = ChatOpenAI(model_name="gpt-4o", callbacks=[logging_handler], temperature=0)
    model_structured = ChatOpenAI(model_name="gpt-4o", callbacks=[logging_handler], temperature=0)

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

    # weather_tool = WeatherInfoTool(weather_config,  tool_manager=tool_manager)

    # current_weather_tool = CurrentWeatherTool(current_weather_config,  tool_manager=tool_manager)

    # search_contacts_tool = SearchContactsTool(search_contacts_config,  tool_manager=tool_manager)

    # send_message_tool = SendMessageTool(send_message_config,  tool_manager=tool_manager)

    # getting tools to use in agent into a function list
    tool_function_list = []

    for t in tool_manager.get_tool_list():

        tool_func = t.get_tool_function()
        tool_function_list.append(tool_func)

    tool_function_seq: Sequence[BaseTool] = tool_function_list

    # Note: this isn't used, the ToolNode handles executing tools
    # tool_executor = ToolExecutor(tools=tool_list)
    # If tool_executor passed in, the tool list is just used
    # to instantiate ToolNode and executor is ignored
    # So all logging and routing should go through the ToolNode

    memory = MemoryCheckpointer()

    agent = KGPlanningInterAgent(
        model_tools=model_tools,
        model_structured=model_structured,
        checkpointer=memory,
        tools=tool_function_seq,
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
