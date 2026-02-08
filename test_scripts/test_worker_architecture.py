#!/usr/bin/env python3
"""
Test script to demonstrate the KGraph worker architecture.

This script shows how workers can be used to dynamically build LangGraph execution graphs
with different types of processing nodes (chat, tool-based, etc.).
"""

from __future__ import annotations

import os
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


import asyncio
import json
from typing import Dict, Any, List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.runtime import get_runtime

# Import agent state and workers
from kgraphplanner.agent.kgraph_base_agent import AgentState
from kgraphplanner.worker.kgraph_chat_worker import KGraphChatWorker
from kgraphplanner.worker.kgraph_tool_worker import KGraphToolWorker

# Import actual tools and tool manager
from kgraphplanner.tool_manager.tool_manager import ToolManager
from vital_agent_kg_utils.vital_agent_rest_resource_client.tools.tool_name_enum import ToolName as ToolNameEnum

load_dotenv()


# Use AgentState from the base agent (matches KGraphToolAgent pattern)


def create_tool_manager():
    """Create a ToolManager instance with real tools."""
    # Create tool manager (no config needed for basic setup)
    tool_manager = ToolManager()
    
    return tool_manager


def create_test_graph() -> StateGraph:
    """Create a test graph with multiple workers to demonstrate multi-instance architecture."""
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Get tool manager
    tool_manager = create_tool_manager()
    
    # Create workers
    chat_worker = KGraphChatWorker(
        name="greeter",
        llm=llm,
        system_directive="You are a friendly assistant. Greet users warmly and offer help."
    )
    
    # Create multiple tool workers with different specializations
    ai_researcher = KGraphToolWorker(
        name="ai_researcher",
        llm=llm,
        system_directive="You are an AI research specialist. Focus on artificial intelligence developments and use web search to find the latest information.",
        tool_manager=tool_manager,
        available_tool_ids=[
            ToolNameEnum.google_web_search_tool.value,
            ToolNameEnum.place_search_tool.value
        ],
        max_iters=2
    )
    
    weather_specialist = KGraphToolWorker(
        name="weather_specialist", 
        llm=llm,
        system_directive="You are a weather information specialist. Use weather tools to provide detailed weather information.",
        tool_manager=tool_manager,
        available_tool_ids=[
            ToolNameEnum.weather_tool.value,
            ToolNameEnum.place_search_tool.value
        ],
        max_iters=2
    )
    
    general_researcher = KGraphToolWorker(
        name="general_researcher",
        llm=llm, 
        system_directive="You are a general research assistant. Use all available tools to provide comprehensive information.",
        tool_manager=tool_manager,
        available_tool_ids=[
            ToolNameEnum.google_web_search_tool.value,
            ToolNameEnum.weather_tool.value,
            ToolNameEnum.place_search_tool.value
        ],
        max_iters=3
    )
    
    # Create the state graph using AgentState
    graph = StateGraph(AgentState)
    
    # Add entry node that creates activation data for multiple workers
    def entry_node(state: AgentState) -> AgentState:
        """Initialize the execution state for multiple workers."""
        messages = state.get("messages", [])
        
        # Get user request from messages or use default
        user_request = "What are the latest developments in artificial intelligence and machine learning in 2024?"
        if messages:
            user_request = messages[-1].content
        
        print(f"🔧 DEBUG: Entry node processing user request: {user_request}")
        
        # Create agent_data structure with activation for multiple worker instances
        agent_data = {
            "activation": {
                "greeter_1": {
                    "prompt": "Greet the user warmly and introduce the research team.",
                    "args": {
                        "message": "Hello! I'm coordinating a team of specialists to help with your request."
                    }
                },
                "ai_researcher_1": {
                    "prompt": "Research AI and ML developments specifically.",
                    "args": {
                        "request": f"Focus on AI/ML aspects: {user_request}",
                        "available_tools": [ToolNameEnum.google_web_search_tool.value, ToolNameEnum.place_search_tool.value],
                        "conversation_history": []
                    }
                },
                "weather_specialist_1": {
                    "prompt": "Provide weather context if relevant to the request.",
                    "args": {
                        "request": "Get current weather for major tech hubs (San Francisco, Seattle, New York)",
                        "available_tools": [ToolNameEnum.weather_tool.value, ToolNameEnum.place_search_tool.value],
                        "conversation_history": []
                    }
                },
                "general_researcher_1": {
                    "prompt": "Provide comprehensive research synthesis.",
                    "args": {
                        "request": f"Synthesize and expand on: {user_request}",
                        "available_tools": [
                            ToolNameEnum.google_web_search_tool.value,
                            ToolNameEnum.weather_tool.value,
                            ToolNameEnum.place_search_tool.value
                        ],
                        "conversation_history": []
                    }
                }
            }
        }
        
        return {
            "messages": messages,
            "agent_data": agent_data
        }
    
    graph.add_node("entry", entry_node)
    graph.add_edge(START, "entry")
    
    # Build worker subgraphs with unique occurrence_ids
    greeter_entry, greeter_exit = chat_worker.build_subgraph(graph, "greeter_1")
    ai_entry, ai_exit = ai_researcher.build_subgraph(graph, "ai_researcher_1")
    weather_entry, weather_exit = weather_specialist.build_subgraph(graph, "weather_specialist_1")
    general_entry, general_exit = general_researcher.build_subgraph(graph, "general_researcher_1")
    
    # Connect the workflow: entry -> greeter -> parallel workers -> end
    graph.add_edge("entry", greeter_entry)
    
    # After greeter, run all researchers in parallel
    graph.add_edge(greeter_exit, ai_entry)
    graph.add_edge(greeter_exit, weather_entry)
    graph.add_edge(greeter_exit, general_entry)
    
    # All workers converge to end
    graph.add_edge(ai_exit, END)
    graph.add_edge(weather_exit, END)
    graph.add_edge(general_exit, END)
    
    return graph


async def run_test():
    """Run the test to demonstrate multiple workers in the same graph."""
    
    print("=== Multi-Worker Architecture Test ===")
    print("Demonstrating multiple worker instances with unique occurrence_ids")
    
    # Create and compile the test graph
    graph = create_test_graph()
    
    print(f"\nTotal nodes in test graph: {len(graph.nodes)}")
    print(f"Node names: {list(graph.nodes.keys())}")
    
    # Compile with checkpointer
    checkpointer = MemorySaver()
    compiled_graph = graph.compile(checkpointer=checkpointer)
    
    print("\nStarting execution with multiple workers...\n")
    
    # Create initial state with a user message
    initial_state = {
        "messages": [HumanMessage(content="What are the latest developments in artificial intelligence and machine learning in 2024?")],
        "agent_data": {}
    }
    
    # Run the graph
    config = {"configurable": {"thread_id": "multi_worker_test"}}
    
    try:
        final_state = await compiled_graph.ainvoke(initial_state, config=config)
        
        print("\n=== Final Results ===")
        
        # Display messages from all workers
        if final_state.get("messages"):
            print("💬 Messages from all workers:")
            for i, msg in enumerate(final_state["messages"]):
                content_preview = msg.content[:150] + "..." if len(msg.content) > 150 else msg.content
                print(f"  {i+1}. {type(msg).__name__}: {content_preview}")
        
        # Display decisions from each worker
        if final_state.get("agent_data") and "decisions" in final_state["agent_data"]:
            print("\n🤖 Worker Decisions:")
            decisions = final_state["agent_data"]["decisions"]
            for worker_id, decision in decisions.items():
                if isinstance(decision, dict) and "answer" in decision:
                    answer_preview = decision["answer"][:100] + "..." if len(decision["answer"]) > 100 else decision["answer"]
                    print(f"  {worker_id}: {answer_preview}")
        
        print("\n✅ Multi-worker architecture test completed successfully!")
        print("\n🎯 This demonstrates multiple worker instances:")
        print("   1. Each worker has unique occurrence_id (greeter_1, ai_researcher_1, etc.)")
        print("   2. Each worker receives targeted activation data")
        print("   3. Workers can run in parallel or sequence")
        print("   4. All workers contribute to the final state")
        print("   5. No data collision between worker instances")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run_test())
