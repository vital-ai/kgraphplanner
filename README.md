# KGraphPlanner

A Python framework for building AI agents that plan and execute multi-step tasks using composable workers, tool integration, and LangGraph-based execution graphs.

## Overview

KGraphPlanner provides a layered architecture for constructing AI agent pipelines:

- **Workers** encapsulate a single capability (chat, tool use, categorization) and generate their own LangGraph subgraphs.
- **Agents** compose workers into end-to-end workflows with state management and checkpointing.
- **Programs & Execution Graphs** let an LLM planner dynamically generate multi-worker pipelines at runtime — including fan-out, fan-in, and conditional routing.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   Agents                        │
│  ChatAgent · ToolAgent · CaseAgent · Planner    │
├─────────────────────────────────────────────────┤
│                   Workers                       │
│  ChatWorker · ToolWorker · CaseWorker           │
├─────────────────────────────────────────────────┤
│              Execution Layer                    │
│  ExecGraphAgent · ProgramSpec · GraphSpec       │
├─────────────────────────────────────────────────┤
│              Infrastructure                     │
│  AgentConfig · ToolManager · Checkpointer       │
└─────────────────────────────────────────────────┘
```

### Agents

| Agent | Purpose |
|-------|---------|
| `KGraphChatAgent` | Single-worker conversational agent with memory |
| `KGraphToolAgent` | Agent with OpenAI function-calling and external tools |
| `KGraphCaseAgent` | Routes requests through LLM-based categorization to specialized workers |
| `KGraphPlannerAgent` | Classifies user requests via LLM and routes to category-specific handlers (greeting, question, planning, etc.) |
| `KGraphExecGraphAgent` | Executes a declarative `GraphSpec` using a registry of workers |

### Workers

| Worker | Purpose |
|--------|---------|
| `KGraphChatWorker` | Single LLM call — responds to a prompt with conversation history |
| `KGraphToolWorker` | Multi-turn tool-calling loop — decides, calls tools, and finalizes |
| `KGraphCaseWorker` | Classifies input into one of N categories via structured LLM output |

### Tools

Built-in tools that connect to a tool server endpoint:

- `google_web_search_tool` — Web search via Google
- `place_search_tool` — Place/business lookup
- `google_address_validation_tool` — Address validation
- `weather_tool` — Weather data

Tools are managed by `ToolManager`, which handles registration, configuration, and JWT authentication.

## Installation

```bash
pip install kgraphplanner
```

Or from source:

```bash
git clone https://github.com/vital-ai/kgraphplanner.git
cd kgraphplanner
pip install -e ".[dev]"
```

### Requirements

- Python >= 3.12
- OpenAI API key
- Tool server running (for tool-based agents)

## Configuration

Configuration is loaded from environment variables with the `KGPLAN__` prefix, using `__` as the hierarchy separator.

### Setup

```bash
cp .env.example .env
# Edit .env with your values
```

### Environment Variables

```bash
# OpenAI
OPENAI_API_KEY=sk-...

# Tool subsystem
KGPLAN__TOOLS__ENDPOINT=http://localhost:8008
KGPLAN__TOOLS__ENABLED=google_web_search_tool,weather_tool

# Model
KGPLAN__AGENT__MODEL__NAME=gpt-4o-mini
KGPLAN__AGENT__MODEL__TEMPERATURE=0.7

# Checkpointing
KGPLAN__CHECKPOINTING__ENABLED=true
KGPLAN__CHECKPOINTING__BACKEND=memory
```

Configuration can also be created programmatically:

```python
from kgraphplanner.config.agent_config import AgentConfig, ToolConfig

# From environment
config = AgentConfig.from_env()

# From code
config = AgentConfig(
    tools=ToolConfig(
        endpoint="http://localhost:8008",
        enabled=["weather_tool", "google_web_search_tool"],
    ),
)

# From YAML
config = AgentConfig.from_yaml("config.yaml")

# From dict
config = AgentConfig.from_dict({"tools": {"endpoint": "..."}})
```

## Quick Start

### Chat Agent

```python
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from kgraphplanner.agent.kgraph_chat_agent import KGraphChatAgent
from kgraphplanner.worker.kgraph_chat_worker import KGraphChatWorker
from kgraphplanner.checkpointer.kgraphmemory_checkpointer import KGraphMemoryCheckpointer
from kgraphplanner.checkpointer.kgraph_serializer import KGraphSerializer

llm = ChatOpenAI(model="gpt-4o-mini")
checkpointer = KGraphMemoryCheckpointer(serde=KGraphSerializer())

worker = KGraphChatWorker(
    name="assistant",
    llm=llm,
    system_directive="You are a helpful assistant.",
    required_inputs=["message"],
)

agent = KGraphChatAgent(name="chat", checkpointer=checkpointer, chat_worker=worker)

async def main():
    config = {"configurable": {"thread_id": "demo"}}
    result = await agent.arun([HumanMessage(content="Hello!")], config=config)
    print(result["messages"][-1].content)

asyncio.run(main())
```

### Case Agent (Intent Routing)

```python
from kgraphplanner.agent.kgraph_case_agent import KGraphCaseAgent
from kgraphplanner.worker.kgraph_case_worker import KGCase

cases = [
    (KGCase(id="weather", name="Weather", description="weather requests"),
     weather_worker),
    (KGCase(id="search", name="Search", description="research requests"),
     search_worker),
]

agent = KGraphCaseAgent(
    name="router",
    case_worker_llm=llm,
    case_worker_pairs=cases,
    resolve_worker=resolve_worker,
)
```

### Tool Agent

```python
from kgraphplanner.agent.kgraph_tool_agent import KGraphToolAgent
from kgraphplanner.config.agent_config import AgentConfig
from kgraphplanner.tool_manager.tool_manager import ToolManager

config = AgentConfig.from_env()
tm = ToolManager(config=config)
tm.load_tools_from_config()

agent = KGraphToolAgent(
    name="tools",
    llm=llm,
    tool_manager=tm,
    available_tool_ids=["weather_tool"],
)
```

## Testing

Tests are organized into modular cases run by dedicated runners:

```bash
# List available test cases
python test_scripts/test_agent_runner.py --list
python test_scripts/test_tool_runner.py --list
python test_scripts/test_case_agent_runner.py --list

# Run all cases in a runner
python test_scripts/test_agent_runner.py

# Run specific cases by name or index
python test_scripts/test_agent_runner.py chat
python test_scripts/test_tool_runner.py 2

# Planner tests
python test_scripts_planner/test_planner_runner.py
```

### Test Runners

| Runner | Cases |
|--------|-------|
| `test_agent_runner.py` | LangGraph agent, tool agents (web search, address, weather, place, multi), chat agent |
| `test_tool_runner.py` | Direct tool tests, tool manager, multi-weather, Times Square |
| `test_case_agent_runner.py` | Case agent routing, case worker categorization |
| `test_kgraphplanner_agent_runner.py` | KGraphPlanner agent |

## Project Structure

```
kgraphplanner/
├── agent/              # Agent implementations
│   ├── kgraph_agent.py           # Abstract interface
│   ├── kgraph_base_agent.py      # Base class with state and checkpointing
│   ├── kgraph_chat_agent.py      # Conversational agent
│   ├── kgraph_tool_agent.py      # Tool-calling agent
│   ├── kgraph_case_agent.py      # Intent-routing agent
│   ├── kgraph_planner_agent.py   # LLM planner agent
│   └── kgraph_exec_graph_agent.py # Declarative graph executor
├── worker/             # Worker implementations
│   ├── kgraph_worker.py          # Abstract base worker
│   ├── kgraph_chat_worker.py     # Chat (single LLM call)
│   ├── kgraph_tool_worker.py     # Tool-calling loop
│   └── kgraph_case_worker.py     # Categorization
├── config/             # Configuration
│   └── agent_config.py           # Typed config with from_yaml/from_dict/from_env
├── tool_manager/       # Tool management
│   └── tool_manager.py           # Registration, config loading, JWT auth
├── tools/              # Built-in tool implementations
├── graph/              # Execution graph models
│   ├── exec_graph.py             # GraphSpec, EdgeSpec, Binding models
│   └── graph_hop.py              # Activation merging utilities
├── program/            # Program specification
│   ├── program.py                # ProgramSpec models
│   └── program_expander.py       # ProgramSpec → GraphSpec expansion
└── checkpointer/       # State persistence
    ├── kgraph_serializer.py      # Custom LangGraph serializer
    └── kgraphmemory_checkpointer.py  # In-memory checkpointer
```

## License

Apache License 2.0