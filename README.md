<h1 align="center">DLens</h1>
<h3 align="center">DeepLense AI Scientist</h3>

<p align="center">
Agentic AI for autonomous scientific workflows in gravitational lensing research
</p>

<div align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white" alt="Python"></a>
  <a href="https://ai.pydantic.dev/"><img src="https://img.shields.io/badge/Pydantic-AI-E92063?logo=pydantic&logoColor=white" alt="Pydantic AI"></a>
  <a href="https://github.com/ML4SCI/DeepLense"><img src="https://img.shields.io/badge/DeepLense-ML4SCI-blueviolet?logo=telescope&logoColor=white" alt="DeepLense"></a>
</div>

<p align="center">
  <a href="./docs/DOCUMENTATION.md"><strong>Technical Documentation</strong></a>
</p>

<p align="center">
  <a href="./docs/AGENT_TRACKER.md"><strong>Agent Tracker</strong></a>
</p>

## Overview

DLens is a multi-agent framework for autonomously orchestrating scientific workflows in gravitational lensing research. It provides agentic capabilities for reasoning, decision-making, and workflow automation, enabling LLM-powered agents to coordinate complex multi-step processes with minimal human supervision.

## Technical Foundation

DLens is built as an abstraction layer over [Pydantic AI](https://ai.pydantic.dev/), leveraging its type-safe agent framework to create robust, composable agentic workflows. This foundation provides structured data validation, seamless integration with language models, and a clean API for building complex multi-agent systems.

## Design Guidelines

DLens is designed to operate primarily with **locally hosted or on-premise language models**, avoiding dependencies on third-party model services. System prompts are **constrained, focused, and information-dense**, minimizing context usage to reduce both computational overhead and the risk of hallucination.

DLens follows a **divide-and-conquer approach** to agent design. Complex tasks are decomposed into smaller, well-scoped subproblems, each handled by a dedicated agent or a single LLM call. This improves reliability, reduces cognitive load on individual models, and enables more predictable and debuggable workflows.

For complex multi-tool-use agents that require larger LLMs with API access, DLens also supports a **ReAct (Reasoning + Acting) framework**, enabling iterative reasoning-action loops where agents can plan, execute tools, observe results, and adapt their strategy across multiple steps.

# Quick Start

This guide walks you through setting up the `dlens` library.

## 1. Install `uv`

`uv` is a fast Python package and environment manager used by this project.

### macOS / Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows

Do yourself a favor and switch to Linux or macOS.

_(But if you insist: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`)_

Restart your terminal, then verify:

```bash
uv --version
```

## 2. Clone the repository

```bash
git clone https://github.com/pranath-reddy/DeepLense-AI-Scientist.git
cd DeepLense-AI-Scientist
```

## 3. Create a virtual environment

```bash
uv venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
```

## 4. Install the package and dependencies

```bash
uv sync            # installs everything, including torch
```

`uv pip install -e .` works too. `torch` is a **main** dependency, not an extra:
the AI-Scientist pipeline's training/inference backends need it, so a plain
`uv sync` is all a fresh clone requires.

**Verified environments** (offline suite + both live pipelines):

| Python | torch | Notes |
|---|---|---|
| 3.12.13 | 2.x | the interpreter the paper results were produced on |
| 3.14.5 | 2.13.0 | also verified end to end; Apple-Silicon MPS detected and used |

`uv` picks the newest interpreter satisfying `requires-python = ">=3.12"`. Pin one
explicitly with `uv venv --python 3.12` if you want to match the paper runs.

## 5. Run the sanity check or pytest

```bash
uv run python scripts/sanity_check.py
```

Or run the test suite:

```bash
uv run pytest
```
