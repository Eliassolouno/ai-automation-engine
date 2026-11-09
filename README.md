# AI Automation Engine (`aae`)

[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://python.org)
[![Architecture](https://img.shields.io/badge/architecture-Hexagonal%20Ports%20%26%20Adapters-emerald.svg)](#architecture)
[![License](https://img.shields.io/badge/license-MIT-purple.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-success.svg)](#testing)
[![Code Style](https://img.shields.io/badge/code%20style-strict%20typed-black.svg)](#tech-stack)

> **Enterprise-grade multi-agent control plane and resilient autonomous execution runtime.**  
> Built for zero vendor lock-in, deterministic step-level replay, strict typed tool contracts, and sub-second execution guardrails.

---

## Architectural Principles

`ai-automation-engine` is designed around strict **Hexagonal Architecture (Ports and Adapters)**:
1. **Zero Provider Lock-in:** The core reasoning kernel knows only `Protocol` contracts (`LLMPort`, `ToolExecutorPort`, `TelemetryPort`). Provider drivers (Anthropic, OpenAI, Local vLLM) live entirely in adapter layers.
2. **Type-Safe Tool Contracts:** All tools are declared via Pydantic models with automated JSON Schema extraction and runtime payload validation.
3. **Budget Guardrails:** Step limits, wall-clock timeouts, and token budgets are strictly enforced at runtime to prevent recursive runaway loops.
4. **Deterministic Observability:** In-memory, JSONL, and OpenTelemetry-compatible tracing capture exact inputs, thoughts, tool calls, and observations for every turn.

```
                      +-----------------------------+
                      |      Multi-Agent App /      |
                      |        FastAPI / CLI        |
                      +--------------+--------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                              AAE KERNEL                                 |
|                                                                         |
|   +------------------+    +-------------------+    +----------------+   |
|   |  AgentLoop       |--->|  BudgetEnforcer   |--->|  RunState      |   |
|   |  (Think-Act-Obs) |    |  (Tokens/Timeout) |    |  (Pydantic V2) |   |
|   +--------+---------+    +-------------------+    +----------------+   |
|            |                                                            |
|       (Ports / Protocols)                                               |
|            +-----------------------+----------------------+             |
+------------|-----------------------|----------------------|-------------+
             v                       v                      v
     +---------------+       +---------------+      +---------------+
     |  LLM Gateway  |       | Tool Registry |      |   Telemetry   |
     |  (Router/API) |       |  & Executor   |      |   & Tracing   |
     +-------+-------+       +-------+-------+      +-------+-------+
             |                       |                      |
             v                       v                      v
       Anthropic /             System Tools /         JSONL Traces /
       OpenAI / vLLM           Custom APIs / MCP        Metrics
```

---

## Directory Structure

```
ai-automation-engine/
├── pyproject.toml              # Modern Hatchling / PEP 621 configuration
├── LICENSE                     # MIT License (Elias Joel Mendoza)
├── README.md                   # Enterprise architecture and quickstart
├── src/aae/
│   ├── types.py                # Core domain types (ToolCall, AgentStep, RunResult)
│   ├── kernel/                 # Pure domain runtime (No external dependencies)
│   │   ├── ports.py            # Hexagonal Protocols (LLMPort, ToolExecutorPort)
│   │   ├── loop.py             # Re-entrant Think -> Act -> Observe execution loop
│   │   └── budget.py           # Guardrails (step limits, token quotas, timeouts)
│   ├── gateway/                # Model abstraction and failover router
│   │   └── router.py           # Multi-provider routing and fallback
│   ├── tools/                  # Type-safe tool definition and execution
│   │   ├── spec.py             # Schema generation from Python signatures
│   │   └── executor.py         # Async/sync execution with timeout bounds
│   └── telemetry/              # Structured traces and latency metering
│       └── tracing.py          # Deterministic step logger
└── tests/
    ├── test_suite.py           # Standard library regression test suite
    └── test_agent_runtime.py   # Pytest async runtime benchmarks
```

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/Eliassolouno/ai-automation-engine.git
cd ai-automation-engine
pip install -e .
```

### 2. Basic Agent with Tool Dispatch

```python
import asyncio
from aae.kernel.loop import AgentLoop
from aae.tools.spec import ToolRegistry
from aae.tools.executor import ToolExecutor
from aae.gateway.router import MockModelGateway
from aae.types import RunBudget

# 1. Register a type-safe tool
registry = ToolRegistry()

@registry.register(name="calculate_tax", description="Computes tax for a given amount")
def calculate_tax(amount: float, rate: float = 0.18) -> float:
    return round(amount * rate, 2)

executor = ToolExecutor(registry)

# 2. Initialize Agent Loop with Model Gateway and Budget
llm = MockModelGateway()
agent = AgentLoop(llm=llm, tools=executor)

# 3. Execute with strict budget guardrails
async def main():
    budget = RunBudget(max_steps=5, max_wallclock_seconds=10.0)
    result = await agent.run("Calculate tax on $1500", budget=budget)
    print(f"Status: {result.status} | Steps: {len(result.steps)}")
    print(f"Final Output: {result.final_answer}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Testing & Quality Gates

Run the automated test suite:

```bash
python tests/test_suite.py
```

Output:
```text
..
----------------------------------------------------------------------
Ran 2 tests in 0.016s

OK
```

---

## Author & License

- **Author:** Elias Joel Mendoza Yzaguirre
- **Email:** `eliasjoelmendozayzaguirre@gmail.com`
- **License:** MIT License. See [LICENSE](LICENSE) for details.
