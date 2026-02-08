from __future__ import annotations

from typing import Dict, Any, Tuple, List, Optional, Literal, Type
from dataclasses import dataclass, field
import json
import asyncio
import time
import logging

logger = logging.getLogger(__name__)

from pydantic import BaseModel, Field, ConfigDict
from langgraph.graph import StateGraph, END
from langgraph.runtime import get_runtime
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.tools import BaseTool

from kgraphplanner.worker.kgraph_worker import KGraphWorker


@dataclass
class KGraphToolWorker(KGraphWorker):
    """
    A tool worker that can decide whether to use tools or generate a final response.
    
    This worker creates a subgraph with:
    - Entry node: decision_node (decides tool vs final)
    - Tool node: tool_node (executes selected tool)
    - Exit node: finalize_node (generates final response)
    
    The decision node loops back to itself after tool execution until
    the worker decides to generate a final response.
    """
    
    tool_manager: Any = None
    available_tool_ids: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize after dataclass creation."""
        super().__post_init__()
        if self.tool_manager is None:
            raise ValueError("tool_manager must be provided to KGraphToolWorker")
        if not self.available_tool_ids:
            raise ValueError("available_tool_ids must be provided and non-empty")
    
    def get_available_tools(self) -> Dict[str, BaseTool]:
        """Get tools that are available to this worker based on tool IDs."""
        tools = {}
        for tool_id in self.available_tool_ids:
            tool = self.tool_manager.get_tool(tool_id)
            if tool:
                tools[tool_id] = tool.get_tool_function()
        return tools
    
    def _make_decision_model(self) -> Type[BaseModel]:
        """Create a Pydantic model for structured decision output."""
        tool_keys = tuple(self.available_tool_ids)
        ToolNameLit = Literal[tool_keys] if tool_keys else Literal["__no_tools__"]
        
        def _require_all_props(schema: dict) -> None:
            """Make all properties required for function calling."""
            props = schema.get("properties")
            if isinstance(props, dict):
                schema["required"] = list(props.keys())
        
        class WorkerDecision(BaseModel):
            model_config = ConfigDict(extra="forbid", json_schema_extra=_require_all_props)
            type: Literal["tool", "final"]
            tool_name: Optional[ToolNameLit] = None
            arguments: Dict[str, Any] = Field(default_factory=dict)
            answer: Optional[str] = None
        
        return WorkerDecision
    
    def build_subgraph(self, graph_builder: StateGraph, occurrence_id: str) -> Tuple[str, str]:
        """
        Build a three-node subgraph for tool-based processing.
        
        Returns:
            Tuple of (entry_node_id, exit_node_id)
        """
        decision_node_id = self._safe_node_id(occurrence_id, "decision")
        tool_node_id = self._safe_node_id(occurrence_id, "tool")
        finalize_node_id = self._safe_node_id(occurrence_id, "finalize")
        
        async def decision_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """Make decisions about tool usage based on the current state."""
            _t0 = time.time()
            runtime = get_runtime()
            writer = runtime.stream_writer
            
            # Get activation data for this worker
            activation = self._get_activation(state, occurrence_id)
            prompt = activation.get("prompt", "")
            args = activation.get("args", {})
            
            logger.info(f"⏱️ [{time.strftime('%H:%M:%S')}] Tool worker '{occurrence_id}' decision_node START")
            logger.debug(f"Tool worker '{occurrence_id}' checking activation: {activation}")
            
            # Check if we have activation data or tool results to work with
            slot = self._get_worker_slot(state, occurrence_id)
            
            # Check for tool results in THIS worker's own tool_history
            tool_history = slot.get("tool_history", [])
            has_tool_results = len(tool_history) > 0
            
            if not prompt and not args and not has_tool_results:
                logger.debug(f"Tool worker '{occurrence_id}' has no activation or tool results, creating final decision")
                # Create a final decision to terminate the worker
                slot["last_decision"] = {
                    "type": "final",
                    "tool_name": None,
                    "arguments": {},
                    "tool_call_id": None,
                    "answer": "No activation data provided"
                }
                return {"work": {occurrence_id: slot}}
            
            # If no activation but we have tool results, generate final answer from results
            if not prompt and not args and has_tool_results:
                logger.debug(f"Tool worker '{occurrence_id}' has no activation but has tool results, generating final answer")
                logger.debug(f"Found {len(recent_tool_messages)} recent tool messages")
                
                # Build messages for final answer generation
                messages = []
                if self.system_directive:
                    messages.append(SystemMessage(content=self.system_directive))
                
                # Add the original user request from state messages
                for msg in state_messages:
                    if hasattr(msg, 'content') and msg.__class__.__name__ == 'HumanMessage':
                        messages.append(msg)
                        break
                
                # Add recent tool results
                for tool_msg in reversed(recent_tool_messages):
                    messages.append(SystemMessage(content=f"Tool result: {tool_msg.content}"))
                
                messages.append(SystemMessage(content="Based on the tool results above, provide a comprehensive final answer to the user's question."))
                
                logger.debug(f"Generating final answer with {len(messages)} messages")
                
                # Call LLM to generate final answer
                try:
                    response = await self.llm.ainvoke(messages)
                    final_answer = response.content if hasattr(response, 'content') else str(response)
                    
                    logger.debug(f"Generated final answer: {final_answer[:100]}...")
                    
                    slot["last_decision"] = {
                        "type": "final",
                        "tool_name": None,
                        "arguments": {},
                        "tool_call_id": None,
                        "answer": final_answer
                    }
                    
                    # Store the final decision in agent_data.decisions for resolve worker to access
                    agent_data = dict(state.get("agent_data", {}))
                    decisions = dict(agent_data.get("decisions", {}))
                    decisions[occurrence_id] = slot["last_decision"]
                    agent_data["decisions"] = decisions
                    
                    logger.debug(f"Stored final decision in agent_data.decisions[{occurrence_id}]")
                    
                    return {
                        "work": {occurrence_id: slot},
                        "agent_data": {"decisions": {occurrence_id: slot["last_decision"]}}
                    }
                except Exception as e:
                    logger.debug(f"Error generating final answer: {e}")
                    slot["last_decision"] = {
                        "type": "final",
                        "tool_name": None,
                        "arguments": {},
                        "tool_call_id": None,
                        "answer": "Error generating final answer from tool results"
                    }
                    
                    return {
                        "work": {occurrence_id: slot},
                        "agent_data": {"decisions": {occurrence_id: slot["last_decision"]}}
                    }
            
            # Get worker slot and check iteration limit
            slot = self._get_worker_slot(state, occurrence_id)
            slot["iters"] += 1
            
            if slot["iters"] > self.max_iters:
                logger.info(
                    f"⏱️ [{time.strftime('%H:%M:%S')}] Tool worker '{occurrence_id}' "
                    f"hit max_iters ({self.max_iters}), making final summarization call"
                )
                # Build messages for a final summarization call (no tools)
                summary_messages = []
                if self.system_directive:
                    summary_messages.append(SystemMessage(content=self.system_directive))
                prompt = activation.get("prompt", "")
                args = activation.get("args", {})
                if prompt:
                    summary_messages.append(SystemMessage(content=f"Task instructions: {prompt}"))
                if args and "request" in args:
                    summary_messages.append(HumanMessage(content=args["request"]))
                elif prompt or args:
                    parts = []
                    if prompt:
                        parts.append(prompt)
                    if args:
                        parts.append(f"Context: {json.dumps(args, default=str)}")
                    summary_messages.append(HumanMessage(content="\n".join(parts)))
                # Include tool history so the LLM can see all gathered data
                tool_history = slot.get("tool_history", [])
                for ai_msg, tool_msgs in tool_history:
                    summary_messages.append(ai_msg)
                    for tool_msg in tool_msgs:
                        summary_messages.append(tool_msg)
                summary_messages.append(SystemMessage(
                    content="You have used all available tool iterations. "
                            "Do NOT call any more tools. Provide your final, "
                            "comprehensive answer now using the information "
                            "you have already gathered above."
                ))
                try:
                    _t_sum = time.time()
                    summary_response = await self.llm.ainvoke(summary_messages)
                    _t_sum_done = time.time()
                    answer = getattr(summary_response, 'content', '') or ''
                    logger.info(
                        f"⏱️ [{time.strftime('%H:%M:%S')}] Tool worker '{occurrence_id}' "
                        f"max-iter summarization took {_t_sum_done - _t_sum:.1f}s  "
                        f"content_len={len(answer)}"
                    )
                except Exception as e:
                    logger.warning(f"Tool worker '{occurrence_id}' max-iter summarization failed: {e}")
                    answer = (
                        "Maximum iterations reached. Based on the tool results "
                        "gathered so far, I was unable to produce a final summary."
                    )
                slot["last_decision"] = {"type": "final", "answer": answer}
                return {
                    "work": {occurrence_id: slot},
                    "agent_data": {
                        "decisions": {occurrence_id: slot["last_decision"]},
                        "errors": {occurrence_id: f"Max iterations exceeded in {occurrence_id}"}
                    }
                }
            
            # Build messages for decision
            messages = []
            if self.system_directive:
                messages.append(SystemMessage(content=self.system_directive))
            
            prompt = activation.get("prompt", "")
            args = activation.get("args", {})
            
            if prompt:
                messages.append(SystemMessage(content=f"Task instructions: {prompt}"))
            
            # Add the user's request as a human message
            logger.debug(f"Args received: {args}")
            if args and "request" in args:
                messages.append(HumanMessage(content=args["request"]))
                logger.debug(f"Added user request from args['request']")
            elif prompt or args:
                # Construct a request from prompt + args when no explicit 'request' key
                parts = []
                if prompt:
                    parts.append(prompt)
                if args:
                    parts.append(f"Context: {json.dumps(args, default=str)}")
                request_text = "\n".join(parts)
                messages.append(HumanMessage(content=request_text))
                logger.debug(f"Constructed user request from prompt+args ({len(request_text)} chars)")
            else:
                logger.debug("No request, prompt, or args available")
            
            messages.append(SystemMessage(content=f"Available tools: {self.available_tool_ids}"))
            
            # Get LangChain tools for binding to LLM
            langchain_tools = []
            for tool_id in self.available_tool_ids:
                tool_obj = self.tool_manager.get_tool(tool_id)
                if tool_obj:
                    tool_function = tool_obj.get_tool_function()
                    langchain_tools.append(tool_function)
            
            # Add guidance for decision making
            if slot["iters"] == 1:
                messages.append(SystemMessage(content="""You have access to tools to help answer questions. Use the available tools when needed, or provide a direct answer if no tools are required.

If you need to use a tool, call the appropriate function with the required parameters.
If you have enough information to answer directly, provide your response without using tools."""))
            else:
                messages.append(SystemMessage(content=f"This is iteration {slot['iters']}. If tools are failing or you have enough information, provide a final answer instead of retrying tools."))
            
            # Add conversation history from THIS worker's own tool_history
            # (isolated per-occurrence to prevent cross-contamination between parallel workers)
            tool_history = slot.get("tool_history", [])
            logger.debug(f"Worker {occurrence_id} tool_history: {len(tool_history)} pairs")
            
            for pair_idx, (ai_msg, tool_msgs) in enumerate(tool_history):
                messages.append(ai_msg)
                for tool_msg in tool_msgs:
                    messages.append(tool_msg)
                logger.debug(f"Added history pair {pair_idx}: AI({len(ai_msg.tool_calls)} calls) + {len(tool_msgs)} ToolMessages")
            
            logger.debug(f"About to call LLM with {len(messages)} messages")
            logger.debug(f"Message types: {[type(m).__name__ for m in messages]}")
            
            # Note: writer functionality removed for simplicity
            
            try:
                # Use OpenAI function calling with tools bound to LLM
                if langchain_tools:
                    llm_with_tools = self.llm.bind_tools(langchain_tools)
                else:
                    llm_with_tools = self.llm
                
                _t_llm = time.time()
                response = await llm_with_tools.ainvoke(messages)
                _t_llm_done = time.time()
                
                _llm_elapsed = _t_llm_done - _t_llm
                raw_content = getattr(response, 'content', None)
                content_len = len(raw_content) if isinstance(raw_content, str) else 0
                tool_calls_count = len(response.tool_calls) if hasattr(response, 'tool_calls') and response.tool_calls else 0

                logger.info(
                    f"⏱️ [{time.strftime('%H:%M:%S')}] Tool worker '{occurrence_id}' "
                    f"LLM call took {_llm_elapsed:.1f}s  content_len={content_len}  "
                    f"tool_calls={tool_calls_count}"
                )

                meta = getattr(response, 'response_metadata', {}) or {}
                usage = meta.get('token_usage') or meta.get('usage', {})
                logger.info(
                    f"📊 Tool worker '{occurrence_id}' "
                    f"finish_reason={meta.get('finish_reason', '?')}  "
                    f"model={meta.get('model_name', meta.get('model', '?'))}  "
                    f"prompt_tokens={usage.get('prompt_tokens', '?')}  "
                    f"completion_tokens={usage.get('completion_tokens', '?')}  "
                    f"total_tokens={usage.get('total_tokens', '?')}"
                )

                if not raw_content and tool_calls_count == 0:
                    logger.warning(
                        f"⚠️ Tool worker '{occurrence_id}' got empty response with no tool calls. "
                        f"raw_content repr: {repr(raw_content)[:500]}"
                    )
                    for attr in ('content', 'additional_kwargs', 'response_metadata',
                                 'tool_calls', 'invalid_tool_calls', 'usage_metadata'):
                        val = getattr(response, attr, '<missing>')
                        logger.warning(f"  response.{attr} = {repr(val)[:300]}")
                
                # Process LLM response and determine action
                decision = self._process_llm_response(response, slot)
                
                # Store decision in work slot
                slot["last_decision"] = decision
                slot["iters"] = slot.get("iters", 0) + 1
                
                # Store AIMessage in worker's own slot (NOT shared state messages)
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    logger.debug(f"Storing AIMessage with tool_calls in worker slot")
                    slot["pending_ai_message"] = response
                
                logger.info(f"⏱️ [{time.strftime('%H:%M:%S')}] Tool worker '{occurrence_id}' decision_node END ({time.time() - _t0:.1f}s)")
                
                # Return partial update — reducers handle merging
                return {
                    "work": {occurrence_id: slot},
                    "agent_data": {"decisions": {occurrence_id: decision}}
                }
                
            except Exception as e:
                # Force final decision on error
                slot["last_decision"] = {"type": "final", "answer": "I encountered an error while processing your request, but I'll do my best to help with the information available."}
                
                return {
                    "work": {occurrence_id: slot},
                    "agent_data": {"errors": {occurrence_id: str(e)}}
                }
        
        async def tool_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """Execute all tool calls from the LLM response."""
            _t0 = time.time()
            logger.info(f"⏱️ [{time.strftime('%H:%M:%S')}] Tool worker '{occurrence_id}' tool_node START")
            
            # Get pending AIMessage from THIS worker's slot (isolated per-occurrence)
            slot = self._get_worker_slot(state, occurrence_id)
            ai_message_with_tools = slot.get("pending_ai_message")
            
            if not ai_message_with_tools:
                logger.debug(f"No pending_ai_message in worker slot for {occurrence_id}")
                return {}
            
            logger.debug(f"Found AIMessage with {len(ai_message_with_tools.tool_calls)} tool calls")
            
            # Execute all tool calls and collect ToolMessages
            tool_messages = []
            
            for i, tool_call in enumerate(ai_message_with_tools.tool_calls):
                tool_call_id = tool_call['id']
                tool_name = tool_call['name']
                tool_args = tool_call.get('args', {})
                
                logger.debug(f"Processing tool call {i+1}/{len(ai_message_with_tools.tool_calls)}: {tool_name} id={tool_call_id} args={tool_args}")
                
                # Validate tool exists
                if tool_name not in self.available_tool_ids:
                    error_msg = f"Tool '{tool_name}' not available. Available tools: {self.available_tool_ids}"
                    tool_message = ToolMessage(
                        content=error_msg,
                        tool_call_id=tool_call_id,
                        name=tool_name
                    )
                    tool_messages.append(tool_message)
                    logger.debug(f"Tool {tool_name} not found, added error message")
                    continue
                
                # Get tool object and execute
                tool_obj = self.tool_manager.get_tool(tool_name)
                if not tool_obj:
                    error_msg = f"Tool '{tool_name}' object not found in manager"
                    tool_message = ToolMessage(
                        content=error_msg,
                        tool_call_id=tool_call_id,
                        name=tool_name
                    )
                    tool_messages.append(tool_message)
                    logger.debug(f"Tool object {tool_name} not found, added error message")
                    continue
                
                try:
                    # Get tool function and execute
                    tool_func = tool_obj.get_tool_function()
                    activation = self._get_activation(state, occurrence_id)
                    
                    # Coerce tool arguments
                    args = self._coerce_tool_args(tool_func, tool_args, activation.get("args", {}))
                    
                    logger.debug(f"Executing tool {tool_name} with coerced args: {args}")
                    _t_tool = time.time()
                    result = await tool_func.ainvoke(args)
                    logger.info(f"⏱️ [{time.strftime('%H:%M:%S')}] Tool '{tool_name}' for '{occurrence_id}' took {time.time() - _t_tool:.1f}s")
                    logger.debug(f"Raw tool result type: {type(result)}")
                    
                    # Convert result to JSON string for tool message content
                    try:
                        if hasattr(result, 'model_dump'):
                            result_json = json.dumps(result.model_dump(), indent=2)
                        elif hasattr(result, 'dict'):
                            result_json = json.dumps(result.dict(), indent=2)
                        else:
                            result_json = json.dumps(str(result), indent=2)
                        result_text = result_json
                    except Exception as json_error:
                        logger.debug(f"JSON serialization failed: {json_error}")
                        result_text = str(result)
                    
                    # Create ToolMessage
                    tool_message = ToolMessage(
                        content=result_text,
                        tool_call_id=tool_call_id,
                        name=tool_name
                    )
                    tool_messages.append(tool_message)
                    logger.debug(f"Created ToolMessage for {tool_name} id={tool_call_id} len={len(result_text)}")
                    
                except Exception as e:
                    error_msg = f"Error executing tool '{tool_name}': {str(e)}"
                    tool_message = ToolMessage(
                        content=error_msg,
                        tool_call_id=tool_call_id,
                        name=tool_name
                    )
                    tool_messages.append(tool_message)
                    logger.debug(f"Tool {tool_name} execution failed: {e}")
            
            # Store completed AI+Tool pair in worker's own tool_history (NOT shared messages)
            if tool_messages:
                history = list(slot.get("tool_history", []))
                history.append((ai_message_with_tools, tool_messages))
                slot["tool_history"] = history
                slot.pop("pending_ai_message", None)
                logger.debug(f"Stored AI+Tool pair in worker slot tool_history ({len(history)} total)")
            
            logger.info(f"⏱️ [{time.strftime('%H:%M:%S')}] Tool worker '{occurrence_id}' tool_node END ({time.time() - _t0:.1f}s)")
            # Return partial update
            return {"work": {occurrence_id: slot}}
        
        def finalize_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """Generate final response and clean up."""
            logger.info(f"⏱️ [{time.strftime('%H:%M:%S')}] Tool worker '{occurrence_id}' finalize_node START")
            
            # Get decision from agent_data (primary location)
            agent_decisions = state.get("agent_data", {}).get("decisions", {})
            decision = agent_decisions.get(occurrence_id, {})
            
            # Fallback to work slot if not found in agent_data
            if not decision:
                slot = self._get_worker_slot(state, occurrence_id)
                decision = slot.get("last_decision", {})
            
            answer = decision.get("answer", "No activation data provided")
            logger.debug(f"Finalize node - answer: {str(answer)[:100] if answer else 'None'}...")
            
            activation = state.get("agent_data", {}).get("activation", {}).get(occurrence_id, {})
            
            logger.info(f"⏱️ [{time.strftime('%H:%M:%S')}] Tool worker '{occurrence_id}' finalize_node END")
            
            # Return partial update — reducers handle merging
            return {
                "agent_data": {
                    "decisions": {occurrence_id: decision},
                    "results": {
                        occurrence_id: {
                            "result_text": answer or "",
                            "args_used": activation.get("args", {})
                        }
                    }
                }
            }
        
        # Add nodes to graph
        graph_builder.add_node(decision_node_id, decision_node)
        graph_builder.add_node(tool_node_id, tool_node)
        graph_builder.add_node(finalize_node_id, finalize_node)
        
        # Add conditional routing from decision node
        def route_decision(state: Dict[str, Any]) -> str:
            logger.debug(f"=== ROUTE_DECISION CALLED ===")
            
            # Always check work slot first for most recent decision
            work = state.get("work", {})
            slot = work.get(occurrence_id, {})
            decision = slot.get("last_decision", {})
            
            # Fallback to agent_data if not in work slot
            if not decision:
                agent_decisions = state.get("agent_data", {}).get("decisions", {})
                decision = agent_decisions.get(occurrence_id, {})
            
            decision_type = decision.get("type")
            route = "TOOL" if decision_type == "tool" else "FINAL"
            logger.debug(f"Routing decision - type: {decision_type}, route: {route}")
            return route
        
        graph_builder.add_conditional_edges(
            decision_node_id,
            route_decision,
            {"TOOL": tool_node_id, "FINAL": finalize_node_id}
        )
        
        # Finalize node must terminate the graph
        graph_builder.add_edge(finalize_node_id, END)
        
        # Tool node loops back to decision
        graph_builder.add_edge(tool_node_id, decision_node_id)
        
        # Don't set entry point - this causes LangGraph to auto-connect to START
        # The entry point will be connected through conditional routing from worker_setup
        
        return decision_node_id, finalize_node_id
    
    def _push_tool_message(self, slot: Dict[str, Any], tool_name: str, result: Any):
        """Add tool result to conversation history."""
        messages = slot.get("messages", [])
        text = json.dumps(result, ensure_ascii=False)[:2000]
        messages.append(SystemMessage(content=f"[TOOL {tool_name}] {text}"))
        slot["messages"] = messages[-6:]  # Keep last 6 messages
    
    def _coerce_tool_args(self, tool: BaseTool, planned: dict, act_args: dict) -> dict:
        """Fill in required args from activation or defaults."""
        args = dict(planned or {})
        schema = getattr(tool, "args_schema", None)
        
        if schema and hasattr(schema, "model_fields"):
            for name, field in schema.model_fields.items():
                if name in args:
                    continue
                
                # Heuristics for common parameter names
                if name == "query":
                    candidate = (
                        act_args.get("query")
                        or act_args.get("company")
                        or act_args.get("topic")
                        or act_args.get("search_term")
                    )
                    if candidate:
                        args["query"] = str(candidate)
                        continue
                
                # Use field default if present
                if field.default is not None:
                    args[name] = field.default
        
        return args
    
    def _process_llm_response(self, response, slot: Dict[str, Any]) -> Dict[str, Any]:
        """Process LLM response and extract decision information."""
        logger.debug(f"Processing LLM response in _process_llm_response")
        
        # Check if response has tool calls (OpenAI function calling)
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_call = response.tool_calls[0]  # Take first tool call
            tool_name = tool_call['name']
            arguments = tool_call['args']
            tool_call_id = tool_call['id']
            
            logger.debug(f"Found tool call - name: {tool_name}, args: {arguments}, id: {tool_call_id}")
            
            decision = {
                "type": "tool",
                "tool_name": tool_name,
                "arguments": arguments,
                "tool_call_id": tool_call_id,
                "answer": None
            }
            
            logger.debug(f"Created tool decision: {decision}")
            return decision
        
        # No tool calls - this is a final answer
        content = response.content if hasattr(response, 'content') else str(response)
        logger.debug(f"No tool calls found, creating final decision with content: {content[:100]}...")
        
        decision = {
            "type": "final",
            "tool_name": None,
            "arguments": {},
            "tool_call_id": None,
            "answer": content
        }
        
        logger.debug(f"Created final decision: {decision}")
        return decision
