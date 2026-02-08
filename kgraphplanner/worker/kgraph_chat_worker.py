from __future__ import annotations

import time
import json
import logging
from typing import Dict, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

from langgraph.graph import StateGraph
from langgraph.runtime import get_runtime
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from kgraphplanner.worker.kgraph_worker import KGraphWorker


@dataclass
class KGraphChatWorker(KGraphWorker):
    """
    A simple chat worker that performs a single LLM call to respond to a message.
    
    This worker creates a subgraph with:
    - Entry node: chat_node (performs LLM call)
    - Exit node: chat_node (same node, single step)
    
    The worker takes the activation prompt and args, makes an LLM call,
    and stores the response in the state.
    """
    
    def build_subgraph(self, graph_builder: StateGraph, occurrence_id: str) -> Tuple[str, str]:
        """
        Build a simple single-node subgraph for chat processing.
        
        Returns:
            Tuple of (entry_node_id, exit_node_id) - both are the same for this simple worker
        """
        chat_node_id = self._safe_node_id(occurrence_id, "chat")
        
        async def chat_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """Perform a single LLM call and return the response."""
            _t0 = time.time()
            runtime = get_runtime()
            writer = runtime.stream_writer
            
            # Get activation data
            activation = self._get_activation(state, occurrence_id)
            prompt = activation.get("prompt", "")
            args = activation.get("args", {})
            
            logger.info(f"⏱️ [{time.strftime('%H:%M:%S')}] Chat worker '{occurrence_id}' START")
            logger.debug(f"Chat worker '{occurrence_id}' checking activation: {activation}")
            
            # Only proceed if we have actual activation data
            if not prompt and not args:
                logger.debug(f"Chat worker '{occurrence_id}' has no activation, skipping")
                return {}
            
            logger.debug(f"Chat worker '{occurrence_id}' processing with prompt: {prompt[:100]}...")
            logger.debug(f"Chat worker '{occurrence_id}' args keys: {list(args.keys()) if args else 'None'}")
            
            # Build messages for LLM
            messages = []
            if self.system_directive:
                messages.append(SystemMessage(content=self.system_directive))
            
            if prompt:
                messages.append(SystemMessage(content=f"Task instructions: {prompt}"))
            
            # Include checkpointed conversation history from state.
            # The checkpointer restores prior messages into state["messages"]
            # so multi-turn memory works across invocations.
            # When state has conversation history the current HumanMessage is
            # already included (added by arun via initial_state), so we must
            # NOT add a duplicate from args.
            state_messages = state.get("messages", [])
            if state_messages:
                messages.extend(state_messages)
                user_query = state_messages[-1].content if state_messages else ""
                logger.debug(f"Chat worker '{occurrence_id}' included {len(state_messages)} history messages from state")
                # When activation args carry data that is NOT already in
                # state_messages (e.g. resolve_worker receiving worker_output
                # from a previous worker step), append it so the LLM can see it.
                if args and "worker_output" in args:
                    wo = str(args["worker_output"])
                    messages.append(HumanMessage(content=wo))
                    user_query = wo
                    logger.debug(f"Chat worker '{occurrence_id}' appended worker_output from activation args ({len(wo)} chars)")
            else:
                # No state messages — fall back to args-based HumanMessage
                if args and "request" in args:
                    user_query = args["request"]
                elif args and any(k in args for k in ("query", "message", "worker_output")):
                    user_query = args.get("query", args.get("message", args.get("worker_output", "")))
                elif prompt or args:
                    # Construct request from prompt + args (same pattern as tool worker)
                    parts = []
                    if prompt:
                        parts.append(prompt)
                    if args:
                        # Format args readably for the LLM
                        for k, v in args.items():
                            if isinstance(v, list):
                                parts.append(f"\n{k}:")
                                for i, item in enumerate(v):
                                    item_str = json.dumps(item, default=str) if not isinstance(item, str) else item
                                    parts.append(f"  [{i}] {item_str[:2000]}")
                            elif isinstance(v, dict):
                                parts.append(f"\n{k}: {json.dumps(v, default=str)[:2000]}")
                            else:
                                parts.append(f"\n{k}: {v}")
                    user_query = "\n".join(parts)
                else:
                    user_query = "Please respond based on the task instructions."
                messages.append(HumanMessage(content=str(user_query)))
            
            logger.debug(f"Chat worker '{occurrence_id}' sending {len(messages)} messages to LLM")
            logger.debug(f"Chat worker '{occurrence_id}' user query length: {len(str(user_query))}")
            
            writer({
                "phase": "chat_start",
                "node": occurrence_id,
                "worker": self.name,
                "activation": activation
            })
            
            try:
                # Make LLM call
                _t_llm = time.time()
                response = await self.llm.ainvoke(messages)
                _elapsed = time.time() - _t_llm

                # --- Comprehensive LLM response logging ---
                raw_content = getattr(response, 'content', None)
                content_type = type(raw_content).__name__
                result_text = raw_content if isinstance(raw_content, str) else str(raw_content) if raw_content else ""

                logger.info(
                    f"⏱️ [{time.strftime('%H:%M:%S')}] Chat worker '{occurrence_id}' "
                    f"LLM call took {_elapsed:.1f}s  content_type={content_type}  "
                    f"content_len={len(result_text)}"
                )

                meta = getattr(response, 'response_metadata', {}) or {}
                usage = meta.get('token_usage') or meta.get('usage', {})
                logger.info(
                    f"� Chat worker '{occurrence_id}' "
                    f"finish_reason={meta.get('finish_reason', '?')}  "
                    f"model={meta.get('model_name', meta.get('model', '?'))}  "
                    f"prompt_tokens={usage.get('prompt_tokens', '?')}  "
                    f"completion_tokens={usage.get('completion_tokens', '?')}  "
                    f"total_tokens={usage.get('total_tokens', '?')}"
                )

                # Log if content is unexpectedly empty or short
                if not result_text or len(result_text) < 10:
                    logger.warning(
                        f"⚠️ Chat worker '{occurrence_id}' got empty/short response "
                        f"(len={len(result_text)}). raw_content repr: {repr(raw_content)[:500]}"
                    )
                    # Dump all response attributes for diagnosis
                    for attr in ('content', 'additional_kwargs', 'response_metadata',
                                 'tool_calls', 'invalid_tool_calls', 'usage_metadata'):
                        val = getattr(response, attr, '<missing>')
                        logger.warning(f"  response.{attr} = {repr(val)[:300]}")
                
                writer({
                    "phase": "chat_complete",
                    "node": occurrence_id,
                    "worker": self.name,
                    "result": result_text[:200] + "..." if len(result_text) > 200 else result_text
                })
                
                decision = {
                    "type": "final",
                    "tool_name": None,
                    "arguments": {},
                    "tool_call_id": None,
                    "answer": result_text
                }
                
                logger.debug(f"Chat worker '{occurrence_id}' stored final decision in agent_data.decisions")
                
                # Return partial update — reducers handle merging.
                # Include the AIMessage in "messages" so it gets checkpointed
                # for conversation continuity across invocations.
                finalize = self._finalize_result(state, occurrence_id, result_text)
                finalize["agent_data"]["decisions"] = {occurrence_id: decision}
                finalize["messages"] = [AIMessage(content=result_text)]
                logger.info(f"⏱️ [{time.strftime('%H:%M:%S')}] Chat worker '{occurrence_id}' END ({time.time() - _t0:.1f}s)")
                return finalize
                
            except Exception as e:
                writer({
                    "phase": "chat_error",
                    "node": occurrence_id,
                    "worker": self.name,
                    "error": str(e)
                })
                
                error_result = f"Chat error: {str(e)}"
                decision = {
                    "type": "final",
                    "tool_name": None,
                    "arguments": {},
                    "tool_call_id": None,
                    "answer": error_result
                }
                
                # Return partial update — reducers handle merging
                finalize = self._finalize_result(state, occurrence_id, error_result)
                finalize["agent_data"]["decisions"] = {occurrence_id: decision}
                return finalize
        
        # Add the single node to the graph
        graph_builder.add_node(chat_node_id, chat_node)
        
        # Return the same node as both entry and exit
        return chat_node_id, chat_node_id
