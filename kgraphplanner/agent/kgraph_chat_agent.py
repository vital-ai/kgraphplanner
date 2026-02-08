from __future__ import annotations

from typing import Dict, Any, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.base import BaseCheckpointSaver

from kgraphplanner.agent.kgraph_base_agent import KGraphBaseAgent, AgentState
from kgraphplanner.worker.kgraph_chat_worker import KGraphChatWorker


class KGraphChatAgent(KGraphBaseAgent):
    """
    A simple chat agent that uses a single KGraphChatWorker to respond to messages.
    
    This agent maintains conversation history through the checkpointer and provides
    a straightforward chat interface using the worker architecture.
    """
    
    def __init__(self, *, name: str, checkpointer=None, chat_worker: KGraphChatWorker):
        super().__init__(name, checkpointer)
        
        # Chat worker is required - either provided or must be set later
        if chat_worker is None:
            raise ValueError("chat_worker must be provided to KGraphChatAgent")
        
        self.chat_worker = chat_worker
    
    def build_graph(self) -> StateGraph:
        """
        Build a simple graph with start -> chat_worker -> end.
        """
        graph = StateGraph(AgentState)
        
        # Add the chat worker's subgraph using worker.name as occurrence_id
        worker_entry, worker_exit = self.chat_worker.build_subgraph(graph, self.chat_worker.name)
        
        # Connect start directly to worker and worker to end
        graph.add_edge(START, worker_entry)
        graph.add_edge(worker_exit, END)
        
        return graph
    
    async def arun(self, messages, config=None):
        """Override to handle the worker-based execution state."""
        compiled_graph = self.get_compiled_graph()
        
        # Set up activation data for the chat worker using standard agent_data pattern
        initial_state = {
            "messages": messages,
            "agent_data": {
                "activation": {
                    self.chat_worker.name: {
                        "prompt": "Respond to the user's message based on the conversation history.",
                        "args": {
                            "message": messages[-1].content if messages else "Hello",
                            "conversation_history": [msg.content for msg in messages[:-1]] if len(messages) > 1 else []
                        }
                    }
                },
                "results": {},
                "errors": {},
                "work": {}
            }
        }
        
        # Execute the graph.
        # The chat worker now returns AIMessage in its state update so it gets
        # checkpointed automatically for conversation continuity.
        result = await compiled_graph.ainvoke(initial_state, config=config)
        return result
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about this chat agent."""
        base_info = super().get_agent_info()
        base_info.update({
            "agent_type": "chat",
            "chat_worker_name": self.chat_worker.name,
            "system_directive": self.chat_worker.system_directive[:100] + "..." if len(self.chat_worker.system_directive) > 100 else self.chat_worker.system_directive
        })
        return base_info
