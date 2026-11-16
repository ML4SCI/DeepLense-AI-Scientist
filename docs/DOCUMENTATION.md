# DLens (DeepLense AI Scientist) Technical Documentation
**Last updated: 2026-06-04**

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [Core Framework Components](#2-core-framework-components)
3. [Agent Workflow Design](#3-agent-workflow-design)
4. [Tooling Layer](#4-tooling-layer)
5. [Model Integration](#5-model-integration)
6. [Design Principles](#6-design-principles)
7. [Code Organization Map](#7-code-organization-map)
8. [Developer Usage Guide](#8-developer-usage-guide)
9. [Known Limitations and TODOs](#9-known-limitations-and-todos)

---

## 1. High-Level Architecture

### Overview

DLens is an agentic framework built as an abstraction layer over [Pydantic AI](https://ai.pydantic.dev/) for designing modular, type-safe AI agents for autonomous scientific workflows.

### Architectural Principles

- **Local-First Inference**: Designed to operate with locally hosted LLMs (Ollama, llama.cpp, vLLM) rather than external API services
- **Type-Safe Design**: Leverages Pydantic for strict input/output validation
- **Divide-and-Conquer**: Complex tasks are decomposed into focused sub-agents
- **Multi-Agent Orchestration**: Composable agent pipelines with typed I/O

### System Dependency Flow

```
User Request
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│                     E2E / Orchestration Agents                   │
│            (Compose multiple sub-agents in workflows)            │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DLensBaseAgent                              │
│  (Wraps Pydantic AI Agent with DLens system prompts)            │
└─────────────────────────────────────────────────────────────────┘
     │
     ├──────────────────┬──────────────────┐
     ▼                  ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Toolsets   │  │   Schemas    │  │    Models    │
│  (MCP/REST)  │  │  (Pydantic)  │  │ (LLM Wrappers│
└──────────────┘  └──────────────┘  └──────────────┘
```

### Key Modules

| Module | Path | Responsibility |
|--------|------|----------------|
| `agents` | `src/dlens/agents/` | Agent base classes and implementations |
| `tools` | `src/dlens/tools/` | Tool abstractions, MCP wrappers, REST clients |
| `memory` | `src/dlens/memory/` | Session stores for conversational agents |
| `core` | `src/dlens/core.py` | Core utilities (health check) |

---

## 2. Core Framework Components

### 2.1 DLensBaseAgent

**Location**: `src/dlens/agents/_base.py`

The primary agent wrapper class that encapsulates Pydantic AI's `Agent` class with DLens-specific configurations and system prompts.

#### Class Definition

```python
class DLensBaseAgent:
    def __init__(
        self,
        *,
        config: BaseAgentConfig,
        deps_type: Any | None = None,
        output_type: Any | None = None,
        register_tools: Callable | None = None,
        **agent_kwargs: Any,
    )
```

#### Key Methods

| Method | Description |
|--------|-------------|
| `arun(query, deps, **kwargs)` | Execute agent on a query, returns structured result |
| `arun_stream(query, deps, **kwargs)` | Stream agent execution with event logging |
| `_arun_stream_events(query, deps, **kwargs)` | Yield raw LLM events for custom streaming handlers |
| `_build_system_prompt(custom_prompt)` | Construct system prompt with DLens context |

#### Usage Example

```python
from dlens.agents import DLensBaseAgent, BaseAgentConfig, OllamaModel

config = BaseAgentConfig(
    name="My Agent",
    description="Processes scientific data and returns structured results",
    model=OllamaModel(model_name="qwen3:8b", port="11434"),
    debug=True
)

agent = DLensBaseAgent(
    config=config,
    output_type=MyOutputSchema,
    retries=3
)

result = await agent.arun(query="Analyze this dataset: {...}")
```

### 2.2 DLensConversationalAgent

**Location**: `src/dlens/agents/_base.py`

Extended base agent with conversation memory support via `InMemorySessionStore`.

```python
agent = DLensConversationalAgent(config=config)
session_id = agent.create_session()

result = await agent.arun(query="Hello", session_id=session_id)
```

### 2.3 BaseAgentConfig

**Location**: `src/dlens/agents/_base.py`

```python
class BaseAgentConfig(BaseModel):
    name: str                           # Agent identifier
    description: str                    # Purpose description (used in system prompt)
    custom_system_prompt: Optional[str] # Additional instructions
    model: Any                          # Model wrapper instance
    max_history_messages: int = 10      # For conversational agents
    debug: bool = False                 # Enable debug logging
```

### 2.4 AbstractBaseAgent

Generic abstract base for workflow agents that follow a typed input/output pattern.

```python
class AbstractBaseAgent[AbstractInputSchema, AbstractOutputSchema]:
    def arun(self, input: AbstractInputSchema) -> AbstractOutputSchema
```

### 2.5 OutputSchema

Base output schema that all agent outputs should inherit from.

```python
class OutputSchema(BaseModel):
    reasoning: str = Field(description="The reasoning process of the agent.")
```

---

## 3. Agent Workflow Design

### 3.1 Agent Instantiation Pattern

1. **Configuration**: Create a `BaseAgentConfig` with model, name, and description
2. **Schema Binding**: Specify output type (Pydantic model) at instantiation
3. **Tool Registration**: Optionally register tools via callback or toolsets
4. **Execution**: Call `arun()` or `arun_stream()` with query

### 3.2 Multi-Agent Orchestration (E2E Agents)

E2E agents orchestrate multiple sub-agents in a defined workflow:

```python
class MyE2EAgent(AbstractBaseAgent[MyInput, MyOutput]):
    def __init__(self, *, input_schema, output_schema):
        super().__init__(input_schema, output_schema)
        self._step1_agent = DLensBaseAgent(config=step1_config, output_type=Step1Schema)
        self._step2_agent = DLensBaseAgent(config=step2_config, output_type=Step2Schema)

    async def arun(self, input: MyInput) -> MyOutput:
        step1 = await self._step1_agent.arun(query=input.data)
        step2 = await self._step2_agent.arun(query=step1.output)
        return self.output_schema(result=step2.output)
```

---

## 4. Tooling Layer

### 4.1 AbstractBaseTool

**Location**: `src/dlens/tools/_base.py`

Base class for deterministic (non-MCP) tools with typed I/O.

### 4.2 MCP Wrapper

**Location**: `src/dlens/tools/_mcp_wrapper.py`

Converts synchronous REST functions into async MCP-compatible tools.

```python
from dlens.tools._mcp_wrapper import mcp_rest_tool

mcp_rest_tool(
    mcp=mcp,
    name="my_tool",
    rest_fn=my_api_function,
    doc="Tool description for the LLM",
)
```

Features:
- Sync-to-async bridging via `asyncio.to_thread()`
- JSON-safe output handling
- Uniform error handling
- Signature copying for Pydantic model generation

### 4.3 RestClient

**Location**: `src/dlens/tools/_rest_client.py`

Reusable HTTP client with connection pooling, retries, and JSON handling.

```python
from dlens.tools import RestClient, get_api_key

client = RestClient()
data = client.get_json(url, params={"key": api_key})
```

### 4.4 Example MCP Server

**Location**: `src/dlens/tools/example_mcp_server.py`

A reference implementation showing how to wrap REST APIs as MCP tools, using NASA's public APIs as an example.

---

## 5. Model Integration

**Location**: `src/dlens/agents/_models.py`

| Model Class | Backend | Default Port |
|-------------|---------|--------------|
| `OllamaModel` | Ollama | 11434 |
| `LlamaCppModel` | llama.cpp server | 8080 |
| `VLLMModel` | vLLM server | 8000 |
| `OutlinesLlamaCppModel` | Outlines + llama.cpp | — |
| `LiteLLMModel` | LiteLLM proxy | 4000 |
| `OpenAIModel` | OpenAI API | — |

All model wrappers inherit from Pydantic AI's `OpenAIChatModel` (for OpenAI-compatible APIs).

The `VLLMModel` and `LiteLLMModel` wrappers use OpenAI-compatible local
endpoints. Constructing either wrapper only configures the provider; it does
not contact a server.

```python
from dlens.agents import LiteLLMModel, VLLMModel

vllm = VLLMModel(model_name="Qwen/Qwen3-8B", port="8000")
litellm = LiteLLMModel(model_name="local-qwen", port="4000")
```

The default endpoints are `http://localhost:8000/v1/` for vLLM and
`http://127.0.0.1:4000/v1/` for LiteLLM. The wrappers have offline contract
tests; live provider checks require a locally running server and should be
performed separately from the default test suite.

---

## 6. Design Principles

- **Single Responsibility**: Each agent/tool has a focused purpose
- **Composition over Inheritance**: E2E agents compose sub-agents
- **Type Safety**: All inputs/outputs validated through Pydantic models
- **Local-First**: Default to local inference, no cloud API dependencies
- **Constrained Prompts**: Information-dense, minimal context usage
- **Debuggability**: Streaming logs, debug mode, reasoning capture

---

## 7. Code Organization Map

```
src/
└── dlens/
    ├── __init__.py              # Package exports
    ├── core.py                  # Core utilities (ping health check)
    │
    ├── agents/
    │   ├── __init__.py          # Agent exports
    │   ├── _base.py             # DLensBaseAgent, DLensConversationalAgent, BaseAgentConfig
    │   └── _models.py           # Model wrappers (Ollama, LlamaCpp, vLLM, etc.)
    │
    ├── memory/
    │   ├── __init__.py
    │   └── session_store.py     # InMemorySessionStore
    │
    ├── tools/
    │   ├── __init__.py          # Tool exports
    │   ├── _base.py             # AbstractBaseTool
    │   ├── _mcp_wrapper.py      # mcp_rest_tool decorator
    │   ├── _rest_client.py      # RestClient, RestClientConfig
    │   ├── example_rest_tools.py    # Example REST tool implementations
    │   └── example_mcp_server.py    # Example MCP server
    │
    └── prompts/
        └── __init__.py

tests/
└── test_smoke.py                # Basic health check test

scripts/                         # Example scripts
├── sanity_check.py
├── simple_agent_run.py
├── simple_agent_stream.py
├── simple_mcp_test.py
└── test_conv_agent.py

Tutorials/
├── ollama-installation-guide.md
└── llama-cpp-guide.md
```

---

## 8. Developer Usage Guide

### 8.1 Creating a New Agent

```python
from dlens.agents import DLensBaseAgent, BaseAgentConfig, OllamaModel, OutputSchema
from pydantic import Field

class MyOutput(OutputSchema):
    answer: str = Field(description="The extracted answer")
    confidence: float = Field(description="Confidence score 0-1")

config = BaseAgentConfig(
    name="My Agent",
    description="Does a specific task",
    model=OllamaModel(model_name="qwen3:8b"),
)

agent = DLensBaseAgent(config=config, output_type=MyOutput, retries=3)
result = await agent.arun(query="...")
```

### 8.2 Adding a New MCP Tool

```python
from mcp.server.fastmcp import FastMCP
from dlens.tools._mcp_wrapper import mcp_rest_tool

mcp = FastMCP("My Server")

def my_api_call(query: str, limit: int = 10) -> dict:
    """Synchronous API function."""
    return make_api_request(url, {"q": query, "limit": limit})

mcp_rest_tool(mcp=mcp, name="my_tool", rest_fn=my_api_call, doc="Tool description")

MY_TOOLSET = FastMCPToolset(mcp)
```

### 8.3 Running Tests

```bash
uv pip install -e .
uv run pytest
```

---

## 9. Known Limitations and TODOs

| Area | Description |
|------|-------------|
| VLLMModel | Live server compatibility depends on the local vLLM deployment |
| LiteLLMModel | Live proxy compatibility depends on the local LiteLLM deployment |
| Parallel Execution | Sub-agents run sequentially; parallel orchestration not implemented |
| Caching | No built-in response caching |
| Monitoring | Relies on loguru; no integrated observability stack |

---

## Appendix: Dependencies

| Package | Purpose |
|---------|---------|
| `pydantic-ai` | Agent framework |
| `pydantic` | Data validation |
| `mcp` | Model Context Protocol |
| `llama-cpp-python` | Local LLM inference |
| `outlines` | Constrained decoding |
| `loguru` | Logging |
| `requests` | HTTP client |
| `xmltodict` | XML parsing |
| `pandas` | Data manipulation |
| `numba` | JIT compilation |
