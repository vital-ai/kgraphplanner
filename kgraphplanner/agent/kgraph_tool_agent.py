from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.base import BaseCheckpointSaver

from kgraphplanner.agent.kgraph_base_agent import KGraphBaseAgent, AgentState
from kgraphplanner.worker.kgraph_tool_worker import KGraphToolWorker
from kgraphplanner.tool_manager.tool_manager import ToolManager


class KGraphToolAgent(KGraphBaseAgent):
    """
    A tool-based agent that uses a KGraphToolWorker to make decisions and execute tools.
    
    This agent can use multiple tools through a tool manager and maintains conversation
    history through the checkpointer.
    """
    
    def __init__(self, *, name: str, checkpointer: Optional[BaseCheckpointSaver] = None, 
                 tool_worker: KGraphToolWorker, tool_manager: ToolManager, 
                 tool_names: List[str]):
        super().__init__(name=name, checkpointer=checkpointer)
        
        # Tool worker is required
        if tool_worker is None:
            raise ValueError("tool_worker must be provided to KGraphToolAgent")
        
        # Tool manager is required
        if tool_manager is None:
            raise ValueError("tool_manager must be provided to KGraphToolAgent")
        
        # Tool names list is required
        if not tool_names:
            raise ValueError("tool_names list must be provided and non-empty")
        
        self.tool_worker = tool_worker
        self.tool_manager = tool_manager
        self.tool_names = tool_names
        
        # Validate that all requested tools exist in the tool manager
        available_tools = tool_manager.get_tool_names()
        missing_tools = [name for name in tool_names if name not in available_tools]
        if missing_tools:
            raise ValueError(f"Tools not found in tool manager: {missing_tools}. Available: {available_tools}")
    
    def build_graph(self) -> StateGraph:
        """
        Build a graph with entry -> tool_worker -> end.
        """
        graph = StateGraph(AgentState)
        
        # Add entry node that prepares the state for the tool worker
        def entry_node(state: AgentState) -> AgentState:
            messages = state.get("messages", [])
            
            # Prepare agent_data with activation for the tool worker
            user_request = messages[-1].content if messages else "Hello"
            logger.debug(f"Agent passing user request: {user_request}")
            
            agent_data = {
                "activation": {
                    self.tool_worker.name: {
                        "prompt": "Analyze the user's request and use appropriate tools to help them.",
                        "args": {
                            "request": user_request,
                            "available_tools": self.tool_names,
                            "conversation_history": [msg.content for msg in messages[:-1]] if len(messages) > 1 else []
                        }
                    }
                },
                "results": {},
                "errors": {},
                "work": {}
            }
            
            return {
                "messages": messages,
                "agent_data": agent_data
            }
        
        graph.add_node("entry", entry_node)
        graph.add_edge(START, "entry")
        
        # Add the tool worker's subgraph using worker.name as occurrence_id
        worker_entry, worker_exit = self.tool_worker.build_subgraph(graph, self.tool_worker.name)
        
        # Connect entry to worker and worker to end
        graph.add_edge("entry", worker_entry)
        graph.add_edge(worker_exit, END)
        
        return graph
    
    async def arun(self, *, messages, config=None):
        """Override to handle the worker-based execution state."""
        compiled_graph = self.get_compiled_graph()
        
        initial_state = {
            "messages": messages,
            "agent_data": {},
            "work": {}
        }
        
        # Execute the graph
        result = await compiled_graph.ainvoke(initial_state, config=config)
        
        # Extract the tool worker's result and add it as an AI message
        # Use occurrence_id (which is worker.name for single-worker agents)
        agent_data = result.get("agent_data", {})
        results = agent_data.get("results", {})
        worker_result = results.get(self.tool_worker.name, {})
        
        if worker_result and "result_text" in worker_result:
            from langchain_core.messages import AIMessage
            response_text = worker_result["result_text"]
            
            # Add the AI response to the messages
            updated_messages = result["messages"] + [AIMessage(content=response_text)]
            result["messages"] = updated_messages
        
        return result
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about this tool agent."""
        base_info = super().get_agent_info()
        base_info.update({
            "agent_type": "tool",
            "tool_worker_name": self.tool_worker.name,
            "available_tools": self.tool_names,
            "tool_count": len(self.tool_names),
            "system_directive": self.tool_worker.system_directive
        })
        return base_info

