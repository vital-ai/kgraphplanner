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
    
    When enable_interrupt is True the worker acts as an orchestrator in a
    chat↔tool loop:
    - Checks args["lookup_result"] from an upstream tool worker.
    - If found → short-circuits with action="continue" (no LLM call).
    - If not found → uses LLM to draft a question, calls
      request_human_input() to pause the graph, and on resume returns
      action="lookup" with the user's reply so the tool worker can retry.
    - On first entry (no lookup_result) → returns action="lookup".
    """
    
    enable_interrupt: bool = False
    parse_json_response: bool = False
    
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
            
            # Resolve directive early: args may override/append via bindings.
            # Must happen before interrupt check so the directive is
            # available in both the orchestration and normal-chat paths.
            effective_directive = self._resolve_system_directive(args)
            
            # --- Interrupt orchestration (opt-in) ---
            if self.enable_interrupt:
                result = await self._handle_interrupt_orchestration(
                    state, occurrence_id, activation, prompt, args,
                    writer, _t0, effective_directive,
                )
                if result is not None:
                    return result
            
            # Only proceed if we have actual activation data
            if not prompt and not args:
                logger.debug(f"Chat worker '{occurrence_id}' has no activation, skipping")
                return {}
            
            logger.debug(f"Chat worker '{occurrence_id}' processing with prompt: {prompt[:100]}...")
            logger.debug(f"Chat worker '{occurrence_id}' args keys: {list(args.keys()) if args else 'None'}")
            
            # Build messages for LLM
            messages = []
            if effective_directive:
                messages.append(SystemMessage(content=effective_directive))
            
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
                # In multi-worker pipelines an upstream worker's output is
                # passed to this worker via activation args.  That data is
                # NOT in state_messages (which only hold the original user
                # conversation).  Detect novel arg values and append them so
                # the LLM can actually see the data it needs to work with.
                # All novel args are included (not just one) because fan-in
                # gather nodes may pass multiple upstream results as separate
                # named args (e.g. analysis_a, analysis_b for an aggregator).
                if args:
                    last_content = state_messages[-1].content if state_messages else ""
                    novel_parts = []
                    for k, v in args.items():
                        v_str = str(v)
                        if v_str == last_content:
                            continue
                        novel_parts.append((k, v_str))
                    if novel_parts:
                        context = "\n\n".join(
                            f"[{k}]\n{v}" for k, v in novel_parts
                        )
                        messages.append(HumanMessage(content=context))
                        user_query = context
                        logger.debug(
                            f"Chat worker '{occurrence_id}' appended {len(novel_parts)} "
                            f"novel activation arg(s) ({len(context)} chars total)"
                        )
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

            # Log LLM parameters for diagnostics (works for both OpenAI and Anthropic)
            _llm_inner = getattr(self.llm, 'bound', self.llm)  # unwrap RunnableBinding
            _llm_params = {
                "model": getattr(_llm_inner, 'model_name', getattr(_llm_inner, 'model', '?')),
                "temperature": getattr(_llm_inner, 'temperature', '?'),
                "max_tokens": getattr(_llm_inner, 'max_tokens', 'None'),
                "n_messages": len(messages),
            }
            logger.info(f"Chat worker '{occurrence_id}' LLM params: {_llm_params}")

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
                # Normalise across OpenAI / Anthropic field names
                prompt_tok = usage.get('prompt_tokens', usage.get('input_tokens', '?'))
                compl_tok = usage.get('completion_tokens', usage.get('output_tokens', '?'))
                total_tok = usage.get('total_tokens', '?')
                if total_tok == '?' and prompt_tok != '?' and compl_tok != '?':
                    total_tok = prompt_tok + compl_tok
                finish = meta.get('finish_reason', meta.get('stop_reason', '?'))
                model_id = meta.get('model_name', meta.get('model', '?'))
                logger.info(
                    f"\U0001f4ca Chat worker '{occurrence_id}' "
                    f"finish_reason={finish}  "
                    f"model={model_id}  "
                    f"prompt_tokens={prompt_tok}  "
                    f"completion_tokens={compl_tok}  "
                    f"total_tokens={total_tok}"
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
                
                # Optional: parse JSON from response and merge into results.
                # This makes fields like "action", "event_code", etc.
                # available as top-level keys for conditional routing and
                # downstream bindings (e.g. result.get('action') == 'tool').
                if self.parse_json_response:
                    parsed = self._try_parse_json(result_text)
                    if parsed is not None:
                        finalize["agent_data"]["results"][occurrence_id].update(parsed)
                        logger.debug(
                            f"Chat worker '{occurrence_id}' parsed JSON response: "
                            f"{list(parsed.keys())}"
                        )
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
    
    def _make_orchestration_result(
        self,
        state: Dict[str, Any],
        occurrence_id: str,
        action: str,
        **extra_fields,
    ) -> Dict[str, Any]:
        """
        Build a finalized result dict for interrupt orchestration.
        
        The result includes an 'action' field that downstream conditional
        routing uses to decide the next node (e.g., "lookup" → tool worker,
        "continue" → assistant_router).
        """
        finalize = self._finalize_result(state, occurrence_id, action)
        result_payload = finalize["agent_data"]["results"][occurrence_id]
        result_payload["action"] = action
        result_payload.update(extra_fields)
        return finalize
    
    async def _handle_interrupt_orchestration(
        self,
        state: Dict[str, Any],
        occurrence_id: str,
        activation: Dict[str, Any],
        prompt: str,
        args: Dict[str, Any],
        writer,
        _t0: float,
        effective_directive: str = "",
    ) -> Dict[str, Any] | None:
        """
        Interrupt orchestration logic for a chat↔tool loop.
        
        Checks args["lookup_result"] and returns one of:
        - action="continue" if the tool found the answer
        - action="lookup" after interrupting to ask the user for info
        - action="lookup" on first entry (no lookup_result yet)
        - None if orchestration does not apply (falls through to normal chat)
        """
        lookup_result = args.get("lookup_result")
        
        # Tool workers store result_text as a string.  If lookup_result is
        # a JSON string, parse it so the dict checks below work.
        # Anthropic models often wrap JSON in markdown fences (```json...```),
        # so strip those before parsing.
        if isinstance(lookup_result, str):
            _stripped = lookup_result.strip()
            if _stripped.startswith("```"):
                # Remove opening fence (```json or ```) and closing fence (```)
                lines = _stripped.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                _stripped = "\n".join(lines).strip()
            try:
                lookup_result = json.loads(_stripped)
            except (json.JSONDecodeError, TypeError):
                pass
        
        # --- Case 1: Tool found the answer → exit the loop ---
        if isinstance(lookup_result, dict) and lookup_result.get("found"):
            logger.info(
                f"⏱️ [{time.strftime('%H:%M:%S')}] Chat worker '{occurrence_id}' "
                f"orchestration: found=True → action=continue"
            )
            writer({
                "phase": "orchestration_continue",
                "node": occurrence_id,
                "worker": self.name,
                "lookup_result": lookup_result,
            })
            result = self._make_orchestration_result(
                state, occurrence_id, "continue", loan=lookup_result
            )
            result["messages"] = [AIMessage(
                content=f"Account located. Proceeding with details."
            )]
            logger.info(
                f"⏱️ [{time.strftime('%H:%M:%S')}] Chat worker '{occurrence_id}' "
                f"END ({time.time() - _t0:.1f}s)"
            )
            return result
        
        # --- Case 2: Tool did not find the answer → ask the user ---
        if isinstance(lookup_result, dict) and not lookup_result.get("found"):
            tried = lookup_result.get("tried", [])
            logger.info(
                f"⏱️ [{time.strftime('%H:%M:%S')}] Chat worker '{occurrence_id}' "
                f"orchestration: found=False, tried={tried} → drafting question"
            )
            
            # Use LLM to draft a natural-language question
            draft_messages = []
            if effective_directive:
                draft_messages.append(SystemMessage(content=effective_directive))
            draft_messages.append(SystemMessage(
                content=(
                    "The customer wants to look up their account but the lookup "
                    "failed. Draft a short, friendly question asking for "
                    "identifying information (loan ID, email, phone, or "
                    "application ID) so we can try again."
                )
            ))
            if tried:
                draft_messages.append(HumanMessage(
                    content=f"We already tried: {', '.join(str(t) for t in tried)}. "
                            f"Ask for something we haven't tried yet."
                ))
            else:
                draft_messages.append(HumanMessage(
                    content="We have no identifying information yet. "
                            "Ask the customer for their loan ID or email."
                ))
            
            response = await self.llm.ainvoke(draft_messages)
            question = getattr(response, 'content', str(response))
            
            writer({
                "phase": "orchestration_interrupt",
                "node": occurrence_id,
                "worker": self.name,
                "question": question,
            })
            
            # Pause the graph — returns when Command(resume=...) is called
            user_reply = self.request_human_input(
                question=question,
                context={"lookup_result": lookup_result, "tried": tried},
            )
            
            logger.info(
                f"⏱️ [{time.strftime('%H:%M:%S')}] Chat worker '{occurrence_id}' "
                f"orchestration: resumed with user reply ({len(str(user_reply))} chars)"
            )
            
            result = self._make_orchestration_result(
                state, occurrence_id, "lookup",
                user_provided_info=user_reply,
                previous_attempts=tried,
            )
            result["messages"] = [AIMessage(content=question)]
            logger.info(
                f"⏱️ [{time.strftime('%H:%M:%S')}] Chat worker '{occurrence_id}' "
                f"END ({time.time() - _t0:.1f}s)"
            )
            return result
        
        # --- Case 3: First entry (no lookup_result yet) → go to tool ---
        if lookup_result is None:
            initial_context = args.get("initial_lookup_result", {})
            logger.info(
                f"⏱️ [{time.strftime('%H:%M:%S')}] Chat worker '{occurrence_id}' "
                f"orchestration: first entry → action=lookup"
            )
            writer({
                "phase": "orchestration_first_entry",
                "node": occurrence_id,
                "worker": self.name,
            })
            result = self._make_orchestration_result(
                state, occurrence_id, "lookup",
                initial_context=initial_context,
            )
            logger.info(
                f"⏱️ [{time.strftime('%H:%M:%S')}] Chat worker '{occurrence_id}' "
                f"END ({time.time() - _t0:.1f}s)"
            )
            return result
        
        # Orchestration did not match — fall through to normal chat
        return None

    @staticmethod
    def _try_parse_json(text: str) -> dict | None:
        """Attempt to parse *text* as JSON, returning a dict or None.

        Handles common LLM quirks:
        - Markdown code fences (``json ... ``)
        - Leading/trailing whitespace
        - Non-dict JSON (arrays, scalars) → returns None
        """
        if not text:
            return None
        stripped = text.strip()
        # Strip markdown code fences
        if stripped.startswith("```"):
            lines = stripped.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            stripped = "\n".join(lines).strip()
        try:
            parsed = json.loads(stripped)
            return parsed if isinstance(parsed, dict) else None
        except (json.JSONDecodeError, ValueError):
            return None
