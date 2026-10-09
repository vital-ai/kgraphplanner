# Fixed-graph pipelines with KGraphExecGraphAgent

`KGraphExecGraphAgent` runs a declarative `GraphSpec`: worker nodes connected by
edges, with bindings that pass data between steps and conditions that route.
This page covers building and running a pipeline, then the v0.2.0 migration
notes.

## A complete example

A pipeline that plans, drafts, validates (redrafting once, with the problems as
revision notes, if there's time), then formats. If formatting fails, a code
fallback produces a canned reply.

```python
from pydantic import BaseModel
from kgraphplanner.agent.kgraph_exec_graph_agent import KGraphExecGraphAgent
from kgraphplanner.graph.exec_graph import (
    GraphSpec, StartNodeSpec, EndNodeSpec, WorkerNodeSpec, EdgeSpec, Binding, RetrySpec,
)
from kgraphplanner.worker.kgraph_chat_worker import KGraphChatWorker
from kgraphplanner.worker.kgraph_function_worker import KGraphFunctionWorker
from kgraphplanner.worker.kgraph_structured_worker import KGraphStructuredWorker

class Plan(BaseModel):
    steps: list[str]

class Explanation(BaseModel):
    headline: str
    body: str

def validate(args):
    problems = check(args["draft"])          # your rules; a list of strings
    return {"ok": not problems, "retry": bool(problems),
            "problems": problems, "draft": args["draft"]}

workers = {
    "plan": KGraphStructuredWorker(name="plan", llm=llm, schema=Plan),
    "content": KGraphChatWorker(name="content", llm=llm, include_conversation=False),
    "validate": KGraphFunctionWorker(name="validate", fn=validate),
    "format": KGraphStructuredWorker(name="format", llm=llm, schema=Explanation),
    "fallback": KGraphFunctionWorker(name="fallback", fn=lambda args: {"result_text": CANNED_REPLY}),
}
retry_once = RetrySpec(max_attempts=2, initial_interval=2, jitter=False)

def b(node, path):
    return [Binding(from_node=node, path=path)]

spec = GraphSpec(
    graph_id="explain",
    nodes=[
        StartNodeSpec(), EndNodeSpec(),
        WorkerNodeSpec(id="plan", worker_name="plan", timeout_seconds=10, retry=retry_once),
        WorkerNodeSpec(id="content", worker_name="content", timeout_seconds=15,
                       retry=retry_once, reentry="reset"),
        WorkerNodeSpec(id="validate", worker_name="validate"),
        WorkerNodeSpec(id="format", worker_name="format", timeout_seconds=8,
                       retry=retry_once, on_error="record"),
        WorkerNodeSpec(id="fallback", worker_name="fallback"),
    ],
    edges=[
        EdgeSpec(source="start", destination="plan", bindings={"case": b("start", "$.case")}),
        EdgeSpec(source="plan", destination="content", bindings={"steps": b("plan", "$.data.steps")}),
        EdgeSpec(source="content", destination="validate",
                 bindings={"draft": b("content", "$.result_text")}),
        # Redraft once, only if there are more than 17 s left, with the
        # problems and the previous draft as revision notes.
        EdgeSpec(source="validate", destination="content",
                 condition="result.get('retry') and run['time_left'] > 17",
                 max_traversals=1, on_exhausted="default",
                 bindings={"steps": b("plan", "$.data.steps"),
                           "revision_notes": b("validate", "$.problems"),
                           "prior_draft": b("validate", "$.draft")}),
        EdgeSpec(source="validate", destination="format", condition="__default__",
                 bindings={"draft": b("validate", "$.draft")}),
        EdgeSpec(source="format", destination="fallback", condition="on_error"),
        EdgeSpec(source="format", destination="end", condition="__default__"),
        EdgeSpec(source="fallback", destination="end"),
    ],
)

agent = KGraphExecGraphAgent(name="explain", graph_spec=spec, worker_registry=workers)
result = await agent.run({"case": case_data}, timeout=40, recursion_limit=30)
result.results["format"]["data"]   # {"headline": ..., "body": ...}
result.errors                      # {} or {node: {"type", "message", "exception", "attempts"}}
result.usage                       # {"calls", "input_tokens", ..., "by_model", "outcomes"}
result.trace                       # ["plan", "content", "validate", "format"]
```

`reentry="reset"` means the redraft sees only that edge's bindings: the plan,
the revision notes and the prior draft, without the first pass's inputs.

The spec is validated when the agent is built. An invalid spec raises
`GraphSpecError`, whose `.errors` lists every problem. One compiled graph is
safe to share across concurrent runs.

## Running

### `await agent.run(start_data, *, timeout=None, deadline=None, grace=1.0, recursion_limit=200, config=None)`

- Runs without a checkpointer, so it needs no `thread_id` and keeps nothing
  between runs. Returns a `RunResult`.
- `start_data` is merged over the start node's `initial_data`, or over
  `initial_data["args"]` when that key exists. Bindings read it with
  `from_node="start"`.
- `timeout` (seconds) or `deadline` (epoch seconds) sets the **run deadline**.
  `GraphSpec.timeout_seconds` is the default. See
  [Deadlines](#deadlines-and-the-grace-window).
- `config` is passed to LangGraph (callbacks, tags, ...). `recursion_limit` is
  set from its own argument.
- **On failure** it raises whatever the run raised, with **`exc.run_result`**
  holding a `RunResult` of everything produced before the failure: results,
  errors (including the failing node), trace, and usage (including the failed
  attempts).

### `await agent.arun(messages=..., config=..., start_data=None, deadline=None, grace=None)`

- Uses the agent's checkpointer, and needs `configurable.thread_id` only when
  one is set. Returns the final state.
- The last `HumanMessage` is copied into `results["start"]["input"]`. Change
  the key with `inject_last_human_as=`, or pass `None` to turn this off.
  `run()` never injects.
- On failure with a checkpointer, `exc.run_result` is built from the last
  checkpoint.

Neither `run` nor `arun` modifies the `config` or `start_data` you pass in.

### `RunResult`

| Field | Contents |
|---|---|
| `results` | `{node_id: result dict}` for every node that ran, plus `start` |
| `errors` | `{node_id: {"type", "message", "exception", "attempts"}}`. `"__run__"` marks a run-level timeout. |
| `usage` | Run totals: `calls`, token counts, `latency_ms`, `by_model`, `outcomes` |
| `usage_log` | Every LLM call this run, keyed `"<node>#<n>"`, in order |
| `run` | `started_at`, `deadline`, `grace` |
| `trace` | Node activations in order |
| `state` | The final LangGraph state |

### Per-run state

Each run starts clean: results (except `start`), activations, traversal
counts, errors, trace, usage and worker scratch slots are reset. This also
applies on a reused checkpointed thread, so a condition can never read a result
from an earlier turn. Only `messages` (the conversation) carries over.

## Routing

### Edges

Each edge connects a worker to another worker or to `end`. Edges from `start`
go to workers and can't have conditions.

A source either has only unconditional out-edges or only conditional ones.
Mixing them is a build error, because both kinds would fire.

Two more build rules:
- A start edge to a node that is also reached from another worker is pruned,
  with a warning. The node waits for its upstream worker instead.
- With no start edges at all, every worker with no incoming edges is wired to
  the start.

### Conditions

Conditions are parsed when the graph is built, so a typo fails at construction
rather than silently evaluating to False.

| Condition | Matches when |
|---|---|
| `__default__` | nothing else matched (at most one per source) |
| `always` / `true` | always |
| `has:<key>` | `result[key]` is truthy |
| `eq:<key>:<value>` | `str(result[key]) == value` |
| `on_error` | the source recorded an error (`on_error="record"`) |
| any other string | a Python expression |

`<key>` can be a dotted path such as `data.action`. Expressions see these names:

- `result`: the source node's result.
- `results`: every node's result this run.
- `traversals`: how many times each conditional edge was taken this run,
  keyed `"source->destination"`.
- `run`: `started_at`, `deadline`, `now` and `time_left` (`inf` with no deadline).

A condition that raises is an `ExecGraphConditionError`. If nothing matches
and there's no `__default__`, the run raises `ExecGraphRoutingError`; the build
logs a warning for sources without a default.

Expressions are `eval`ed, which is not a sandbox. Build agents for
LLM-generated specs with `allow_expressions=False`, as `KGraphPlannerAgent`
does. `validate_graph_spec(spec, allow_expressions=False)` checks a spec the
same way.

### Loops

`max_traversals` limits how many times a conditional edge is taken **per run**.
`on_exhausted` decides what happens at the limit:

| `on_exhausted` | At the limit |
|---|---|
| `"default"` | take the source's `__default__` edge |
| `"error"` | raise `ExecGraphLoopExhausted` |
| `"source_final_turn"` | re-run the source with a "compose your final answer" prompt (for chat ↔ tool loops) |
| unset | `source_final_turn` for an LLM worker source; `error` for anything else |

`WorkerNodeSpec.reentry` controls what a node sees when it runs again:
- `"keep"` (the default): new bindings are merged into the previous pass's
  inputs. `append_list` and `concat_text` bindings grow on every pass.
- `"reset"`: each pass gets only the node defaults plus the incoming edge's
  bindings.

`EdgeSpec.merge` is unrelated. It only controls how the edge's `prompt`
combines with the destination's existing prompt: `"accumulate"` appends it, and
the other two values replace it.

## Failures, timeouts and retries

Workers raise typed errors from `kgraphplanner.worker.worker_errors`:
`WorkerLLMError` (`llm_error`), `WorkerParseError` (`parse_error`),
`WorkerTimeoutError` (`timeout`), and `WorkerError` (`worker_error`, e.g.
missing inputs). Any other exception is reported as `worker_error`.

Per node, in `WorkerNodeSpec`:

- **`timeout_seconds`**: the budget for each attempt, capped by the run
  deadline (see below).
- **`retry=RetrySpec(...)`**
  - Defaults: `max_attempts=2` (including the first), `initial_interval=0.5`,
    `backoff_factor=2.0`, `max_interval=30`, `jitter=True`. Jitter adds up to
    25%, and never takes a delay past `max_interval`.
  - `retry_on` values:

    | Value | Matches |
    |---|---|
    | `timeout` * | the node's own budget, or an SDK request timeout |
    | `rate_limit` * | HTTP 429 |
    | `server_error` * | 5xx |
    | `connection` * | network failures |
    | `client_error` | 4xx other than 401, 403 and 429 |
    | `auth_error` | 401, 403 |
    | `parse_error` | the reply didn't validate |
    | `llm_error` | any failed LLM call |

    \* in the default `retry_on`.
  - Each retry recomputes the time left. A backoff that would pass the
    deadline isn't taken.
- **`on_error`**, applied after retries:
  - `"raise"` (the default): the exception propagates out of `run`/`ainvoke`.
  - `"record"`: writes `errors[node]` and `results[node] = {"status": "error",
    "error_type", "error" (the message), "result_text": "", "attempts"}`, then
    continues. Route on it with an `on_error`, `eq:status:error` or
    `has:error_type` edge; the build warns if a recording node has none.
    This also works when the node runs in parallel with others: in a fan-out,
    one failed branch is recorded, and a fan-in still runs once with the
    other branches' results.

Interrupts (`interrupt()` / `Command(resume=...)`) pass through timeouts,
retries and `record` unchanged.

Timeout, retry and `on_error` apply to every LangGraph node a worker adds,
custom workers included. A node function that takes LangGraph-injected
parameters other than `config` (such as `writer` or `store`) is left unwrapped,
and a warning is logged.

### Deadlines and the grace window

The deadline stops model calls; deterministic code can still finish the
response.

- **LLM workers** get `min(timeout_seconds, time left)`. Once the deadline has
  passed they fail at once with `WorkerTimeoutError` ("run deadline already
  passed"), without calling the model.
- **Other workers** (function nodes) may also use the **grace window** after
  the deadline: `grace`, default 1 s. So with `on_error="record"`, a timed-out
  step can still route to a code fallback that produces a canned reply.
- **`run()` has a backstop:** if the run is still going shortly after
  deadline + grace, it is cancelled and the builtin `TimeoutError` is raised,
  with `run_result` attached.
- **Sync function nodes run in a thread whenever a budget applies,** so their
  timeout fires on time. A timed-out sync node is **abandoned, not
  cancelled**:
  - Its thread keeps running to completion after the run has timed out, and
    any side effects it has still happen.
  - Enough hung sync nodes can exhaust the default thread pool.

  Keep sync code fast, or make it `async`, which is cancelled properly.

## Workers

| Worker | Use for |
|---|---|
| `KGraphFunctionWorker(fn=...)` | Deterministic code. See below. |
| `KGraphStructuredWorker(schema=Model)` | One LLM call returning a validated pydantic model. See below. |
| `KGraphChatWorker` | One LLM call returning text. |
| `KGraphCaseWorker` | Classify input into one of N cases. |
| `KGraphToolWorker` | Tool-calling loop. |

**`KGraphFunctionWorker(name, fn, allow_reserved_keys=False)`**
- `fn(args)` or `fn(args, ctx)`, sync or async. It returns the result dict.
- `args` is a copy of the node's inputs.
- `ctx` has `node_id`, `prompt`, `results` (a read-only copy) and `run`.
- Returning a non-dict, or any of the reserved keys `status`, `error_type`,
  `error` and `usage`, raises `WorkerError` unless `allow_reserved_keys=True`.

**`KGraphStructuredWorker(name, llm, schema, method=None, result_text_field=None, include_conversation=False)`**
- Stores `results[node] = {"data": model_dump, "result_text", "usage"}`. Bind
  fields with `$.data.<field>`.
- `result_text` is the JSON of `data`, or the value of `result_text_field`.
- A reply that doesn't validate raises `WorkerParseError`.
- With Anthropic models it uses Claude's native structured outputs
  (`method="json_schema"`), because LangChain's default forces a tool call,
  which current Claude models reject. Override with `method=`.
- Kwargs bound with `llm.bind(...)` that are model fields (e.g. `max_tokens`)
  are kept.

**`KGraphChatWorker` options**
- **`include_conversation`** (default True): sends `state["messages"]` and
  appends the reply to it. Set it to False for pipeline steps, so each step
  sees only its own inputs.
- **`parse_json_response`**: parses the reply as a JSON object.
  - The parsed dict is stored under `results[node]["data"]`, and its keys are
    also merged into the result for `result.get('action')` conditions and
    `$.key` bindings.
  - Reserved keys (`result_text`, `args_used`, `data`, `status`, `error_type`,
    `error`, `usage`, `attempts`) are never overwritten.
  - A reply that isn't JSON logs a warning. Add `parse_json_required=True` to
    raise `WorkerParseError` instead. For JSON you depend on, prefer
    `KGraphStructuredWorker`.

**`KGraphToolWorker`**
- When it hits `max_iters`, it asks the model for a final summary. That's still
  an answer, so it's recorded as `errors[node] = {"type": "max_iterations", ...}`
  without setting `status: "error"`.

`WorkerNodeSpec` fields the agent uses: `worker_name`, `defaults`,
`system_directive` (overrides the worker's directive for this node), `on_error`,
`timeout_seconds`, `retry` and `reentry`. `tools`, `required_inputs` and
`max_iterations` are ignored, and a warning is logged if they're set; configure
them on the worker.

### What the model sees

A node's **inputs** are its prompt and args, built when the node is activated:
- **Prompt:** the incoming edge's `prompt`, else the node's
  `defaults["prompt"]`.
- **Args:** the node defaults' args, then the edge's bindings, in that order.
  With `reentry="keep"`, the previous pass's args come first.

**`KGraphChatWorker`**, in order:
1. **System:** the directive. That's the node's or worker's
   `system_directive`, replaced by a bound `system_directive` arg and extended
   by `system_directive_append`. Segmented directives become content blocks;
   see Prompt caching.
2. **System:** `Task instructions: <prompt>`, if there's a prompt.
3. **With the conversation** (`include_conversation=True` and `state["messages"]`
   not empty):
   - the conversation messages;
   - then one user message holding every arg whose value differs from the
     last message, formatted as `[<key>]\n<value>` blocks separated by blank
     lines.
4. **Otherwise,** one user message:
   - **Only one arg sent:** if `request` is among the args, the user message
     is just its value. Failing that, it's the first of `query`, `message` or
     `worker_output` that's present. All other args are **not sent**.
   - **Otherwise:** the prompt, then each arg on its own line. A string is
     written as `<key>: <value>`. A list becomes `<key>:` followed by
     `  [i] <item>` lines. A dict becomes `<key>: <JSON>`.
   - **Truncated:** list items and dicts are cut to 2,000 characters each.
   - **With no prompt and no args,** the worker raises `WorkerError`.

**`KGraphStructuredWorker`**, in order:
1. **System:** the directive, built as for the chat worker.
2. **System:** `Task instructions: <prompt>`, if there's a prompt.
3. **The conversation,** only with `include_conversation=True`.
4. **One user message:**
   - `args["request"]` if present.
   - Otherwise each arg as `<key>: <value>`, with lists and dicts as JSON and
     nothing truncated.
   - Args equal to the last conversation message are dropped. With nothing
     left and a conversation present, no user message is added.
5. **The schema:** passed through structured output, not as prompt text.

**`KGraphCaseWorker`**:
- **System:** the directive.
- **System:** a categorization prompt listing the cases and quoting the input.
  The input is the first of `query`, `message` or `input` in the args.
- **System:** `Additional instructions: <prompt>`, if there's a prompt.
- **User:** `Categorize: <input>`.

The `system_directive` and `system_directive_append` keys are never sent as
args.

### Per-request system prompts

Binding `system_directive` (which replaces the prompt) or
`system_directive_append` (which appends to it) into a node's args overrides
that worker's system prompt for one run:

```python
EdgeSpec(source="start", destination="plan", bindings={
    "system_directive": [Binding(from_node="start", path="$.plan_prompt")],
})
```

The override also applies when a loop re-enters the node. For a fixed
per-node prompt, set `WorkerNodeSpec(system_directive=...)` instead; a binding
still overrides it.

### Prompt caching (Anthropic)

A `system_directive` can be a list of segments:

```python
system_directive=[
    {"text": STABLE_VOICE_GUIDE, "cache": True},
    {"text": "Per-request notes"},
]
```

With a `ChatAnthropic` model, cached segments get
`cache_control: {"type": "ephemeral"}`. With other models, the segments are
joined as plain text.

## Usage, logging and privacy

Every LLM call, **including failed attempts**, is recorded with `model`,
`input_tokens`, `output_tokens`, `reasoning_tokens`, `cache_read_tokens`,
`cache_creation_tokens`, `latency_ms` and an `outcome`:
- `ok`, or
- the error type of a failed attempt: `timeout`, `llm_error` or `parse_error`.

A call that never returned (a rate limit, or a timeout) has zero token counts.
A parse failure did return, so it has its real counts.

It's recorded in three places:
- **`results[node]["usage"]`:** the node's successful call(s). The tool worker
  sums all of its calls.
- **`agent_data.usage_log["<node>#<n>"]`:** every call, in order.
- **`agent_data.usage_total`:** totals with `by_model` and `outcomes`
  breakdowns. `RunResult.usage` returns this.

Failures before any call (missing inputs, a deadline that has already passed)
aren't counted.

By default, request and response content is never logged, at any level, and
never put in stream events. Logs carry sizes, keys and metadata only. To change
this, set these on a worker:

- `log_content=True`: log prompts, args and replies.
- `stream_payloads=True`: include content in custom stream events.
- `record_args=True`: keep a copy of the args as `results[node]["args_used"]`.

## Testing

`kgraphplanner.testing` has offline helpers. For the example above:

```python
from kgraphplanner.testing import ScriptedLLM, assert_node_order, rate_limit_error

llm = ScriptedLLM(script={
    "plan": [{"steps": ["context", "decision"]}],          # structured output
    "content": [rate_limit_error(), "first draft", "second draft"],
    "format": [{"headline": "Approved", "body": "..."}],
})
# ... build workers with llm and check() so that "first draft" fails validation
result = await agent.run({"case": case_data}, timeout=40)
assert_node_order(result, ["plan", "content", "validate", "content", "validate", "format"])
llm.calls_for("content")   # the messages each content call received
```

`ScriptedLLM`:
- **`script`:** responses per spec node, in call order. A response can be:
  - a str (the reply text),
  - a dict or pydantic model (JSON text, or the structured result),
  - an `AIMessage` (e.g. with `tool_calls`),
  - an exception, which is raised,
  - `Delay(seconds, then)`,
  - a callable `(messages) -> response`.
- **`default`:** serves nodes not in the script. Those calls are still logged
  under the spec node id.
- **Inspecting:** `calls` lists every call; `calls_for(node)` lists one node's.
- **`reset()`:** clears the cursors and the call log. They persist across runs
  otherwise.
- **Tokens:** `input_tokens` and `output_tokens` set the fake usage of each
  call.

Also available:
- `rate_limit_error()`: an exception the retry policy treats as `rate_limit`.
- `node_trace(result)`: the node activation order.
- `assert_node_order(result, expected, exact=False)`: checks `expected` as a
  subsequence.

---

## Migrating to v0.2.0

These changes alter behaviour that existing specs may rely on:

1. **No first-branch fallback.** When no condition matches and there's no
   `__default__`, the run raises `ExecGraphRoutingError`. It used to take the
   first branch silently. Add a `__default__` edge wherever a source's
   conditions might all be false.
2. **Conditional edges to `end` route to `end`.** They used to be dropped.
   Separately, unconditional edges to `end` from a source that also has
   conditional edges are now a build error, because both would fire.
3. **Condition errors raise.** A condition that fails to parse is a build
   error. One that raises at runtime is an `ExecGraphConditionError`; it used
   to count as False.
4. **Specs are validated in the constructor.** `GraphSpecError` (a
   `ValueError`) covers:
   - edges that would be dropped,
   - conditions on start edges,
   - mixed routing,
   - `on_exhausted="default"` without a default,
   - `source_final_turn` on a non-LLM source,
   - registry keys that don't match the worker's `name`.
5. **Built-in workers raise on failure.**
   - Chat, case and tool workers raise `WorkerLLMError` instead of storing
     "Chat error: ..." text, `selected_case_id="error"`, or an apology answer.
     That includes the tool worker's final-answer and max-iterations summary
     calls.
   - A chat, case or tool worker that is activated with no inputs (no prompt,
     args or tool results) raises `WorkerError`, instead of writing nothing or
     placeholder text.
   - A worker that was never activated still does nothing, as before. For
     example, `KGraphCaseAgent` wires every candidate worker but activates
     only the selected one.
   - Inside the exec graph agent, set `on_error="record"` on the node to keep
     going.
   - `KGraphChatAgent`, `KGraphCaseAgent` and `KGraphToolAgent` now propagate
     these errors out of `arun`.
6. **Loop limits are per run.** The `max_traversals` count used to be shared
   by every run on the compiled graph. With `on_exhausted` unset, a non-LLM
   source raises `ExecGraphLoopExhausted` at the limit instead of looping.
7. **All per-run state is reset at the start of each run,** including on a
   reused checkpointed thread:
   - results (except `start`),
   - activations,
   - traversal counts,
   - errors,
   - trace,
   - usage,
   - worker scratch slots.

   Only messages carry over. Results from an earlier turn used to stay
   visible, and activations used to accumulate.
8. **Node inputs no longer share objects with the spec or with other runs.**
   List defaults and `append_list` bindings used to be mutated in place, so
   inputs grew across runs and the spec itself changed.
9. **`args_used` is off by default.** Set `record_args=True` on a worker to keep
   it.
10. **Logging.**
    - Content that was logged at INFO/DEBUG is gone unless `log_content=True`:
      activations, tool arguments and results, the tool worker's final
      output, and the dump of short replies.
    - Custom stream events no longer include `activation`, `result`,
      `selected_case`, `lookup_result` or `question` unless
      `stream_payloads=True`. Events carry `arg_keys`, `result_len` and
      `selected_case_id` instead.
11. **`parse_json_response`** still merges parsed keys into the result, except
    reserved keys, and now also stores the dict under `data`.
12. **`_resolve_system_directive` no longer removes keys from `args`.** Custom
    workers should use `_split_directive(args)`, which returns
    `(directive, clean_args)`.
13. **New spec fields.** Old specs still load.
    - `EdgeSpec`: `on_exhausted`.
    - `WorkerNodeSpec`: `on_error`, `timeout_seconds`, `retry` and `reentry`.
    - `WorkerNodeSpec.system_directive` is now honoured.
    - `GraphSpec.timeout_seconds` is now read, as the default run deadline.
14. **`KGraphPlannerAgent`:**
    - Generated specs are built with `allow_expressions=False`.
    - `ConditionalBranch.condition` only accepts declarative forms.
    - An invalid generated graph is re-planned with the errors fed back.
    - New: `execute(graph_spec, start_data, **run_kwargs)`, and `node_defaults`
      (also on `expand_program_to_graph`).
15. **`KGraphSerializer` implements LangGraph's typed serializer protocol.**
    - `dumps_typed` returns `("kgraph+<type>", bytes)` instead of a
      `KGraphSerialized` object.
    - Reloading a thread with `KGraphMemoryCheckpointer` or
      `KGraphRedisCheckpointer` failed since v0.1.3, and works again.
    - Values are encoded by LangGraph's `JsonPlusSerializer`, so messages come
      back as messages (they used to come back as dicts). Tuples come back as
      lists.
    - Values in the old format still load.
16. **The tool worker sends system messages only at the start of a request.**
    Its two summary calls get the tool results as text instead of tool-call
    messages. Claude rejected the old order ("multiple non-consecutive system
    messages").
17. **`KGraphToolWorker.max_iters` means what it says.** The worker counted
    each decision twice, so `max_iters=N` allowed only about N/2 decisions. It
    now allows N decision calls, then makes one summarization call. Workers
    tuned around the old behaviour may make more calls; halve `max_iters` to
    keep the old limit.
