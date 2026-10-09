from __future__ import annotations

from typing import Dict, Any, Tuple, List, Optional, Literal, Type
from dataclasses import dataclass, field
import hashlib
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
from kgraphplanner.worker.llm_usage import extract_usage, sum_usage, usage_state_update
from kgraphplanner.worker.error_info import describe_error
from kgraphplanner.worker.worker_errors import WorkerError, WorkerLLMError


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
    summarization_llm: Any = None  # Optional separate LLM for summarization (e.g. low reasoning)
    summarize_results: bool = True  # When False, skip per-result summarization and preserve raw output
    
    def __post_init__(self):
        """Initialize after dataclass creation."""
        super().__post_init__()
        if self.tool_manager is None:
            raise ValueError("tool_manager must be provided to KGraphToolWorker")
        if not self.available_tool_ids:
            raise ValueError("available_tool_ids must be provided and non-empty")
        if self.summarization_llm is None:
            self.summarization_llm = self.llm
    
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
            if self.log_content:
                logger.debug(f"Tool worker '{occurrence_id}' checking activation: {activation}")
            
            # Resolve directive early: args may override/append via bindings.
            effective_directive, args = self._split_directive(args)
            
            # Check if we have activation data or tool results to work with
            slot = self._get_worker_slot(state, occurrence_id)
            
            # Detect new activation: if the activation args changed since
            # last time, reset iters and tool_history so each orchestrator →
            # tool_executor round trip gets a fresh iteration budget.
            activation_key = hashlib.md5(
                json.dumps(args, sort_keys=True, default=str).encode()
            ).hexdigest()
            if slot.get("_activation_key") != activation_key:
                logger.debug(
                    f"Tool worker '{occurrence_id}' new activation detected "
                    f"(key {slot.get('_activation_key', 'None')} → {activation_key}), "
                    f"resetting iters from {slot.get('iters', 0)}"
                )
                slot["iters"] = 0
                slot["tool_history"] = []
                slot["_summaries"] = {}
                slot["_usage"] = []
                slot.pop("pending_ai_message", None)
                slot["_activation_key"] = activation_key
            
            # Check for tool results in THIS worker's own tool_history
            tool_history = slot.get("tool_history", [])
            has_tool_results = len(tool_history) > 0
            
            if not prompt and not args and not has_tool_results:
                if not self._is_activated(state, occurrence_id):
                    # Not selected (e.g. a KGraphCaseAgent candidate): finish
                    # without an LLM call, as before.
                    slot["last_decision"] = {
                        "type": "final",
                        "tool_name": None,
                        "arguments": {},
                        "tool_call_id": None,
                        "answer": "No activation data provided"
                    }
                    return {"work": {occurrence_id: slot}}
                # Activated with nothing to work on: finishing with
                # placeholder text would look like an answer.
                raise WorkerError(occurrence_id, "no prompt, args or tool results to act on")
            
            # If no activation but we have tool results, generate final answer from results
            if not prompt and not args and has_tool_results:
                logger.debug(f"Tool worker '{occurrence_id}' has no activation but has tool results, generating final answer")
                
                # Build messages for final answer generation. System messages
                # come first (Anthropic rejects them mid-conversation), and the
                # tool results go in as text: this call binds no tools.
                messages = self._directive_messages(effective_directive)
                messages.append(SystemMessage(content=(
                    "Based on the tool results provided, give a comprehensive final "
                    "answer to the user's question.")))
                
                # Add the original user request from state messages
                for msg in state.get("messages", []):
                    if msg.__class__.__name__ == 'HumanMessage':
                        messages.append(msg)
                        break
                
                messages.append(HumanMessage(content=self._tool_results_text(tool_history)))
                
                logger.debug(f"Generating final answer with {len(messages)} messages")
                
                try:
                    _t_sum = time.time()
                    response = await self.summarization_llm.ainvoke(messages)
                except Exception as e:
                    raise WorkerLLMError(
                        occurrence_id, f"final-answer LLM call failed: {describe_error(e, log_content=self.log_content)}",
                        cause=e) from e
                usage = extract_usage(response, latency_ms=(time.time() - _t_sum) * 1000,
                                      llm=self.summarization_llm)
                slot.setdefault("_usage", []).append(usage)
                final_answer = response.content if hasattr(response, 'content') else str(response)
                
                if self.log_content:
                    logger.debug(f"Generated final answer: {final_answer}")
                
                slot["last_decision"] = {
                    "type": "final",
                    "tool_name": None,
                    "arguments": {},
                    "tool_call_id": None,
                    "answer": final_answer
                }
                return {
                    "work": {occurrence_id: slot},
                    "agent_data": {
                        "decisions": {occurrence_id: slot["last_decision"]},
                        **usage_state_update(state, occurrence_id, [usage]),
                    }
                }
            
            # Check iteration limit. Keep using the slot from above: a
            # re-fetch returns a fresh dict on the first activation and
            # loses _activation_key, so the next decision wiped the tool
            # history and summaries and re-ran the tools.
            # iters counts decision calls: the Nth decision runs with
            # iters == N, so max_iters decisions are allowed before the
            # final summarization call.
            slot["iters"] += 1
            
            if slot["iters"] > self.max_iters:
                logger.info(
                    f"⏱️ [{time.strftime('%H:%M:%S')}] Tool worker '{occurrence_id}' "
                    f"hit max_iters ({self.max_iters}), making final summarization call"
                )
                # Build messages for a final summarization call (no tools).
                # System messages first, then the request, then the gathered
                # tool results as text: tool-call messages would need the
                # tools bound, which this call deliberately doesn't do.
                summary_messages = self._directive_messages(effective_directive)
                if prompt:
                    summary_messages.append(SystemMessage(content=f"Task instructions: {prompt}"))
                summary_messages.append(SystemMessage(
                    content="You have used all available tool iterations. "
                            "Do NOT call any more tools. Provide your final, "
                            "comprehensive answer now using the tool results "
                            "already gathered."
                ))
                if args and "request" in args:
                    summary_messages.append(HumanMessage(content=args["request"]))
                elif prompt or args:
                    parts = []
                    if prompt:
                        parts.append(prompt)
                    if args:
                        parts.append(f"Context: {json.dumps(args, default=str)}")
                    summary_messages.append(HumanMessage(content="\n".join(parts)))
                # Include the gathered tool results so the LLM can use them
                summary_messages.append(HumanMessage(
                    content=self._tool_results_text(slot.get("tool_history", []))))
                try:
                    _t_sum = time.time()
                    summary_response = await self.summarization_llm.ainvoke(summary_messages)
                    _t_sum_done = time.time()
                except Exception as e:
                    raise WorkerLLMError(
                        occurrence_id,
                        f"max-iterations summary LLM call failed: {describe_error(e, log_content=self.log_content)}",
                        cause=e) from e
                summary_usage = [extract_usage(
                    summary_response, latency_ms=(_t_sum_done - _t_sum) * 1000,
                    llm=self.summarization_llm)]
                slot.setdefault("_usage", []).extend(summary_usage)
                answer = getattr(summary_response, 'content', '') or ''
                logger.info(
                    f"⏱️ [{time.strftime('%H:%M:%S')}] Tool worker '{occurrence_id}' "
                    f"max-iter summarization took {_t_sum_done - _t_sum:.1f}s  "
                    f"content_len={len(answer)}"
                )
                slot["last_decision"] = {"type": "final", "answer": answer}
                return {
                    "work": {occurrence_id: slot},
                    "agent_data": {
                        "decisions": {occurrence_id: slot["last_decision"]},
                        # Not a failure (there is still an answer), but recorded
                        # in the same shape as errors so it shows up in run().errors.
                        "errors": {occurrence_id: {
                            "type": "max_iterations",
                            "message": f"Max iterations ({self.max_iters}) exceeded in {occurrence_id}",
                            "exception": None,
                            "attempts": 1,
                        }},
                        **usage_state_update(state, occurrence_id, summary_usage),
                    }
                }
            
            # Build messages for decision. All system messages go first:
            # Anthropic rejects system messages after the conversation starts.
            messages = self._directive_messages(effective_directive)
            
            if prompt:
                messages.append(SystemMessage(content=f"Task instructions: {prompt}"))
            messages.append(SystemMessage(content=f"Available tools: {self.available_tool_ids}"))
            
            # Guidance for decision making
            if slot["iters"] == 1:
                messages.append(SystemMessage(content="Execute ONLY the tool call(s) specified in the user's request. Do NOT call any tools beyond what was explicitly requested."))
            elif self.summarize_results:
                messages.append(SystemMessage(content=(
                    f"Iteration {slot['iters']}. "
                    "If all requested tools have returned results, respond with ONLY a JSON object listing the tool_call_ids to include in the final answer, e.g.:\n"
                    '{"include_summaries": ["call_abc", "call_def"]}\n'
                    "Do NOT restate or summarize the results yourself. If tools are still pending, make the remaining calls."
                )))
            else:
                messages.append(SystemMessage(content=(
                    f"Iteration {slot['iters']}. "
                    "If all requested tools have returned results, compose your final answer using "
                    "the tool results in this conversation. If tools are still pending, make the remaining calls."
                )))
            
            # Add the user's request as a human message.
            # Strip orchestrator-internal fields so the tool worker only
            # sees tool-specific data (tool, query, requests).
            _STRIP_KEYS = {"action", "plan", "result_text", "args_used", "answer", "previous_plan", "data", "usage"}
            
            def _clean_tool_args(raw_args: dict) -> dict:
                cleaned = {}
                for k, v in raw_args.items():
                    if k in _STRIP_KEYS:
                        continue
                    # If the value is a JSON string, try to parse and clean it too
                    if isinstance(v, str):
                        try:
                            parsed = json.loads(v)
                            if isinstance(parsed, dict):
                                parsed = {pk: pv for pk, pv in parsed.items() if pk not in _STRIP_KEYS}
                                cleaned[k] = json.dumps(parsed, default=str)
                                continue
                        except (json.JSONDecodeError, TypeError):
                            pass
                    cleaned[k] = v
                return cleaned
            
            if self.log_content:
                logger.debug(f"Args received: {args}")
            if args and "request" in args:
                messages.append(HumanMessage(content=args["request"]))
                logger.debug(f"Added user request from args['request']")
            elif prompt or args:
                clean_args = _clean_tool_args(args) if args else {}
                request_text = json.dumps(clean_args, default=str) if clean_args else ""
                if request_text:
                    messages.append(HumanMessage(content=request_text))
                    logger.debug(f"Constructed user request from cleaned args ({len(request_text)} chars)")
                elif prompt:
                    messages.append(HumanMessage(content=prompt))
                    logger.debug(f"Constructed user request from prompt ({len(prompt)} chars)")
            else:
                logger.debug("No request, prompt, or args available")
            
            # Get LangChain tools for binding to LLM
            langchain_tools = []
            for tool_id in self.available_tool_ids:
                tool_obj = self.tool_manager.get_tool(tool_id)
                if tool_obj:
                    tool_function = tool_obj.get_tool_function()
                    langchain_tools.append(tool_function)
            
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
            except Exception as e:
                raise WorkerLLMError(occurrence_id, f"LLM call failed: {describe_error(e, log_content=self.log_content)}",
                                     cause=e) from e

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
            ctd = usage.get('completion_tokens_details') or {}
            reasoning_tok = ctd.get('reasoning_tokens', '?') if isinstance(ctd, dict) else '?'
            logger.info(
                f"📊 Tool worker '{occurrence_id}' "
                f"finish_reason={meta.get('finish_reason', '?')}  "
                f"model={meta.get('model_name', meta.get('model', '?'))}  "
                f"prompt_tokens={usage.get('prompt_tokens', '?')}  "
                f"completion_tokens={usage.get('completion_tokens', '?')}  "
                f"reasoning_tokens={reasoning_tok}  "
                f"total_tokens={usage.get('total_tokens', '?')}"
            )

            if not raw_content and tool_calls_count == 0:
                logger.warning(
                    f"⚠️ Tool worker '{occurrence_id}' got empty response with no tool calls "
                    f"(finish_reason={meta.get('finish_reason', '?')})"
                )
                if self.log_content:
                    for attr in ('content', 'additional_kwargs', 'response_metadata',
                                 'tool_calls', 'invalid_tool_calls', 'usage_metadata'):
                        val = getattr(response, attr, '<missing>')
                        logger.warning(f"  response.{attr} = {repr(val)}")
            
            # Process LLM response and determine action
            decision = self._process_llm_response(response, slot)
            
            # Store decision in work slot
            slot["last_decision"] = decision
            
            # Store AIMessage in worker's own slot (NOT shared state messages)
            if hasattr(response, 'tool_calls') and response.tool_calls:
                logger.debug(f"Storing AIMessage with tool_calls in worker slot")
                slot["pending_ai_message"] = response
            
            logger.info(f"⏱️ [{time.strftime('%H:%M:%S')}] Tool worker '{occurrence_id}' decision_node END ({time.time() - _t0:.1f}s)")
            
            # Return partial update — reducers handle merging
            usage = extract_usage(response, latency_ms=_llm_elapsed * 1000, llm=self.llm)
            slot.setdefault("_usage", []).append(usage)
            return {
                "work": {occurrence_id: slot},
                "agent_data": {
                    "decisions": {occurrence_id: decision},
                    **usage_state_update(state, occurrence_id, [usage]),
                }
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
            
            # Execute all tool calls in parallel using asyncio.gather
            tool_calls = ai_message_with_tools.tool_calls
            activation = self._get_activation(state, occurrence_id)
            n_calls = len(tool_calls)
            
            # Shared dict for per-result summaries (written by _execute_one)
            summaries: Dict[str, str] = {}
            summary_usages: List[Dict[str, Any]] = []

            async def _execute_one(i: int, tool_call: dict) -> ToolMessage:
                """Execute a single tool call, summarize the result, and return a lightweight ToolMessage."""
                tool_call_id = tool_call['id']
                tool_name = tool_call['name']
                tool_args = tool_call.get('args', {})

                logger.info(f"🔧 Tool call {i+1}/{n_calls} for '{occurrence_id}': {tool_name} "
                            + (f"args={tool_args}" if self.log_content else f"arg_keys={sorted(tool_args)}"))

                # Validate tool exists
                if tool_name not in self.available_tool_ids:
                    error_msg = f"Tool '{tool_name}' not available. Available tools: {self.available_tool_ids}"
                    logger.debug(f"Tool {tool_name} not found, added error message")
                    return ToolMessage(content=error_msg, tool_call_id=tool_call_id, name=tool_name)

                tool_obj = self.tool_manager.get_tool(tool_name)
                if not tool_obj:
                    error_msg = f"Tool '{tool_name}' object not found in manager"
                    logger.debug(f"Tool object {tool_name} not found, added error message")
                    return ToolMessage(content=error_msg, tool_call_id=tool_call_id, name=tool_name)

                try:
                    tool_func = tool_obj.get_tool_function()
                    args = self._coerce_tool_args(tool_func, tool_args, activation.get("args", {}))

                    if self.log_content:
                        logger.info(f"🔍 [{time.strftime('%H:%M:%S')}] Tool '{tool_name}' for '{occurrence_id}' query: {args}")
                    _t_tool = time.time()
                    result = await tool_func.ainvoke(args)
                    _t_tool_elapsed = time.time() - _t_tool
                    logger.info(
                        f"⏱️ [{time.strftime('%H:%M:%S')}] Tool '{tool_name}' for '{occurrence_id}' "
                        f"took {_t_tool_elapsed:.1f}s"
                        + (f"  result:\n{result}" if self.log_content else "")
                    )

                    # Convert result to string via compact_dump() when available.
                    try:
                        if hasattr(result, 'compact_dump'):
                            compact = result.compact_dump()
                            if isinstance(compact, str):
                                result_text = compact
                            else:
                                result_text = json.dumps(compact, indent=2)
                        elif hasattr(result, 'model_dump'):
                            result_text = json.dumps(result.model_dump(), indent=2)
                        elif hasattr(result, 'dict'):
                            result_text = json.dumps(result.dict(), indent=2)
                        else:
                            result_text = json.dumps(str(result), indent=2)
                    except Exception as json_error:
                        logger.debug(f"JSON serialization failed: {json_error}")
                        result_text = str(result)

                    # --- Per-result summarization via summarization_llm ---
                    summary_text = None
                    if not self.summarize_results:
                        summary_text = result_text
                        logger.debug(f"Skipping summarization for '{tool_name}' (summarize_results=False)")
                    else:
                        try:
                            _t_sum = time.time()
                            sum_msgs = [
                                SystemMessage(content="Summarize the following tool result into a concise, natural-language answer. Return ONLY the summary."),
                                HumanMessage(content=result_text),
                            ]
                            sum_resp = await self.summarization_llm.ainvoke(sum_msgs)
                            summary_text = getattr(sum_resp, 'content', '') or ''
                            _sum_elapsed = time.time() - _t_sum
                            summary_usages.append(extract_usage(
                                sum_resp, latency_ms=_sum_elapsed * 1000,
                                llm=self.summarization_llm))
                            
                            meta = getattr(sum_resp, 'response_metadata', {}) or {}
                            usage = meta.get('token_usage') or meta.get('usage', {})
                            ctd = usage.get('completion_tokens_details') or {}
                            reasoning_tok = ctd.get('reasoning_tokens', '?') if isinstance(ctd, dict) else '?'
                            logger.info(
                                f"📝 [{time.strftime('%H:%M:%S')}] Summarized '{tool_name}' for '{occurrence_id}' "
                                f"in {_sum_elapsed:.1f}s  summary_len={len(summary_text)}  "
                                f"prompt_tokens={usage.get('prompt_tokens', '?')}  "
                                f"completion_tokens={usage.get('completion_tokens', '?')}  "
                                f"reasoning_tokens={reasoning_tok}"
                            )
                        except Exception as sum_err:
                            logger.warning(f"Summarization failed for '{tool_name}': "
                                           f"{describe_error(sum_err, log_content=self.log_content)}")
                            summary_text = result_text  # fallback: use raw compact data
                    
                    # Store the full summary by tool_call_id, labeled with tool + args
                    _label_keys = ('query', 'search_query', 'location', 'city')
                    query_label = next((tool_args[k] for k in _label_keys if tool_args.get(k)), None)
                    if not query_label:
                        # Compact repr of all args (e.g. lat/lon)
                        query_label = ", ".join(f"{k}={v}" for k, v in tool_args.items()) or None
                    label = f"[{tool_name}: {query_label}]" if query_label else f"[{tool_name}]"
                    summaries[tool_call_id] = f"{label}\n{summary_text}"
                    
                    # Return ToolMessage with the pre-generated summary.
                    # This is smaller than the raw compact_dump and was
                    # produced by the low-reasoning summarization_llm.
                    logger.debug(f"Created summary ToolMessage for {tool_name} id={tool_call_id} len={len(summary_text)}")
                    return ToolMessage(content=summary_text, tool_call_id=tool_call_id, name=tool_name)

                except Exception as e:
                    error_msg = f"Error executing tool '{tool_name}': {str(e)}"
                    # error_msg goes to the model (it needs the reason to
                    # recover); the log follows log_content.
                    logger.debug(f"Tool {tool_name} execution failed: "
                                 f"{describe_error(e, log_content=self.log_content)}")
                    return ToolMessage(content=error_msg, tool_call_id=tool_call_id, name=tool_name)

            # Launch all tool calls concurrently
            logger.info(f"🚀 [{time.strftime('%H:%M:%S')}] Launching {n_calls} tool call(s) in parallel for '{occurrence_id}'")
            tool_messages = list(await asyncio.gather(
                *(_execute_one(i, tc) for i, tc in enumerate(tool_calls))
            ))
            logger.info(f"✅ [{time.strftime('%H:%M:%S')}] All {n_calls} tool call(s) completed for '{occurrence_id}'")
            
            # Store completed AI+Tool pair in worker's own tool_history (NOT shared messages)
            if tool_messages:
                history = list(slot.get("tool_history", []))
                history.append((ai_message_with_tools, tool_messages))
                slot["tool_history"] = history
                slot.pop("pending_ai_message", None)
                logger.debug(f"Stored AI+Tool pair in worker slot tool_history ({len(history)} total)")
            
            # Merge per-result summaries into the slot for assembly at finalize time
            existing = slot.get("_summaries", {})
            existing.update(summaries)
            slot["_summaries"] = existing
            
            logger.info(f"⏱️ [{time.strftime('%H:%M:%S')}] Tool worker '{occurrence_id}' tool_node END ({time.time() - _t0:.1f}s)")
            # Return partial update
            slot.setdefault("_usage", []).extend(summary_usages)
            update: Dict[str, Any] = {"work": {occurrence_id: slot}}
            usage_update = usage_state_update(state, occurrence_id, summary_usages)
            if usage_update:
                update["agent_data"] = usage_update
            return update
        
        def finalize_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """Generate final response and clean up."""
            logger.info(f"⏱️ [{time.strftime('%H:%M:%S')}] Tool worker '{occurrence_id}' finalize_node START")
            
            # Prefer work slot's last_decision (always current within the
            # worker's sequential decision→tool→decision loop) over
            # agent_data.decisions which can be stale when multiple workers
            # run in parallel (e.g. KGraphCaseAgent fan-out).
            slot = self._get_worker_slot(state, occurrence_id)
            decision = slot.get("last_decision", {})
            
            # Fallback to agent_data if work slot has no decision
            if not decision:
                agent_decisions = state.get("agent_data", {}).get("decisions", {})
                decision = agent_decisions.get(occurrence_id, {})
            
            answer = decision.get("answer", "No activation data provided")
            if not isinstance(answer, str):
                answer = json.dumps(answer, indent=2, default=str)
            
            activation = state.get("agent_data", {}).get("activation", {}).get(occurrence_id, {})
            
            # Render the final output with proper line breaks
            rendered = (answer or "").replace("\\n", "\n")
            sep = "=" * 72
            lines = [
                f"⏱️ [{time.strftime('%H:%M:%S')}] Tool worker '{occurrence_id}' finalize_node END",
                sep,
                f"  FINAL OUTPUT: {occurrence_id}",
                sep,
            ]
            if self.log_content:
                for line in rendered.split("\n"):
                    lines.append(f"  {line}")
            else:
                lines.append(f"  ({len(rendered)} chars; set log_content=True to log the output)")
            lines.append(sep)
            logger.info("\n".join(lines))
            
            result: Dict[str, Any] = {"result_text": answer or ""}
            if slot.get("_usage"):
                result["usage"] = sum_usage(slot["_usage"])
            if self.record_args:
                result["args_used"] = activation.get("args", {})
            # Return partial update — reducers handle merging
            return {
                "agent_data": {
                    "decisions": {occurrence_id: decision},
                    "results": {occurrence_id: result}
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
        
        # Tool node loops back to decision
        graph_builder.add_edge(tool_node_id, decision_node_id)
        
        # NOTE: No finalize_node → END edge here.  The exec_graph_agent
        # adds END edges for nodes listed in GraphSpec.exit_points, matching
        # how KGraphChatWorker and KGraphCaseWorker work.  This allows the
        # tool worker to be used as an intermediate node in a pipeline
        # (e.g. a chat worker ↔ tool worker loop).
        
        return decision_node_id, finalize_node_id
    
    @staticmethod
    def _tool_results_text(tool_history) -> str:
        """The tool results in ``tool_history`` as one text block, for LLM
        calls that don't bind tools (and so can't take tool-call messages)."""
        parts = []
        for _ai_msg, tool_msgs in tool_history:
            for tool_msg in tool_msgs:
                name = getattr(tool_msg, "name", None) or "tool"
                parts.append(f"[{name}]\n{tool_msg.content}")
        return "Tool results:\n\n" + ("\n\n".join(parts) if parts else "(none)")

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
            
            logger.debug(f"Found tool call - name: {tool_name}, id: {tool_call_id}")
            
            decision = {
                "type": "tool",
                "tool_name": tool_name,
                "arguments": arguments,
                "tool_call_id": tool_call_id,
                "answer": None
            }
            
            if self.log_content:
                logger.debug(f"Created tool decision: {decision}")
            return decision
        
        # No tool calls - this is a final answer.
        # Try to parse the LLM's content as JSON with include_summaries
        # refs, and assemble from stored per-result summaries.
        raw = response.content if hasattr(response, 'content') else str(response)
        stored = slot.get("_summaries", {})

        if not self.summarize_results:
            # summarize_results=False: the LLM was asked to compose its own
            # answer from the raw tool results — use it directly.
            content = raw
            logger.debug(f"Using LLM-composed answer ({len(raw)} chars; summarize_results=False)")
        elif stored:
            # summarize_results=True: the LLM should have returned a JSON
            # object with include_summaries refs.  Parse and assemble.
            try:
                parsed = json.loads(raw.strip().strip('`').removeprefix('json').strip())
                refs = parsed.get("include_summaries", [])
                if refs:
                    picked = [stored[r] for r in refs if r in stored]
                    if picked:
                        content = "\n\n".join(picked)
                        logger.debug(f"Assembled final answer from {len(picked)}/{len(refs)} referenced summaries")
                    else:
                        content = "\n\n".join(stored.values())
                        logger.debug(f"No matching refs; using all {len(stored)} stored summaries")
                else:
                    # LLM returned valid JSON without include_summaries.
                    # Preserve the raw JSON — the worker may need structured
                    # output (e.g. {"found": true, …}) for downstream routing.
                    content = raw
                    logger.debug(f"LLM returned valid JSON without include_summaries; preserving raw response ({len(raw)} chars)")
            except (json.JSONDecodeError, ValueError):
                # Unexpected: LLM didn't follow the include_summaries
                # guidance — fall back to all stored summaries.
                content = "\n\n".join(stored.values())
                logger.warning(f"Could not parse JSON refs; falling back to all {len(stored)} stored summaries")
        else:
            content = raw
        logger.debug(f"Creating final decision with content len={len(content)}")
        
        decision = {
            "type": "final",
            "tool_name": None,
            "arguments": {},
            "tool_call_id": None,
            "answer": content
        }
        
        if self.log_content:
            logger.debug(f"Created final decision: {decision}")
        return decision
