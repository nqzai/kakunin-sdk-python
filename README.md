# kakunin

[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/nqzai/kakunin-sdk-python/badge)](https://scorecard.dev/viewer/?uri=github.com/nqzai/kakunin-sdk-python)

Python SDK for the [Kakunin](https://kakunin.ai) AI agent compliance API. Issues X.509 certificates to AI agents, monitors behavioral baselines, and enforces scope limits at the tool layer.

```bash
pip install kakunin
```

Requires Python 3.9+.

---

## Quick Start

```python
import asyncio
from kakunin import Kakunin

async def main():
    async with Kakunin(api_key="kak_live_...") as client:
        # Register an agent
        agent = await client.agents.create(
            name="TradeBot-1",
            model="gpt-4o",
            version="2024-11",
        )

        # Record a behavioral event — returns risk_band
        event = await client.events.create(
            agent_id=agent.id,
            action_type="transaction_initiated",
            details={"amount_usd": 50000, "venue": "NYSE"},
        )
        print(event.risk_band)  # "low" | "medium" | "high"

asyncio.run(main())
```

---

## verify_agent_scope Decorator

Wrap any function (sync or async) to verify that the agent is active and holds required scopes before execution. Raises `ScopeViolationError` on failure — never swallows it.

```python
from kakunin import Kakunin, verify_agent_scope

client = Kakunin(api_key="kak_live_...")

@verify_agent_scope(client, agent_id="agt-123", required_scopes=["trade.execute"])
async def execute_trade(order: dict) -> dict:
    # Only runs if agent agt-123 is active and has trade.execute scope
    return await broker.submit(order)

# Also works on sync functions
@verify_agent_scope(client, agent_id="agt-123", required_scopes=["data.read"])
def fetch_prices(symbol: str) -> list[float]:
    return market_data.get(symbol)
```

---

## LangChain Integration

### Option A — KakuninToolGuard (per-tool scope enforcement)

Wraps any LangChain tool. Scope is verified before every `_run` / `_arun` call.

```python
from langchain_core.tools import tool
from kakunin import Kakunin
from kakunin.integrations.langchain import KakuninToolGuard

client = Kakunin(api_key="kak_live_...")

@tool
def execute_trade(order: str) -> str:
    """Execute a trade order on the exchange."""
    return broker.submit(order)

# Wrap the tool — drop into any LangChain agent as-is
guarded_trade = KakuninToolGuard(
    kakunin=client,
    agent_id="agt-123",
    tool=execute_trade,
    required_scopes=["trade.execute"],
)

from langchain.agents import create_react_agent
agent = create_react_agent(llm, tools=[guarded_trade], prompt=prompt)
```

### Option B — verify_agent_scope on tool functions

```python
from langchain_core.tools import tool
from kakunin import Kakunin, verify_agent_scope

client = Kakunin(api_key="kak_live_...")

@tool
@verify_agent_scope(client, agent_id="agt-123", required_scopes=["trade.execute"])
async def execute_trade(order: str) -> str:
    """Execute a trade order."""
    return await broker.submit(order)
```

### Option C — Chain-level guard via callback

Guards an entire LangChain chain, not individual tools.

```python
from kakunin.integrations.langchain import langchain_scope_callback

guard = langchain_scope_callback(
    client, agent_id="agt-123", required_scopes=["trade.execute"]
)
chain = my_chain.with_config(callbacks=[guard])
```

---

## AutoGen Integration

`KakuninConversableAgent` subclasses AutoGen's `ConversableAgent`. It emits behavioral events for every message and verifies scope before each reply.

```python
from autogen import UserProxyAgent
from kakunin import Kakunin
from kakunin.integrations.autogen import KakuninConversableAgent

client = Kakunin(api_key="kak_live_...")

risk_agent = KakuninConversableAgent(
    kakunin=client,
    agent_id="agt-456",
    required_scopes=["risk.assess"],   # verified before every generate_reply()
    # Standard AutoGen kwargs:
    name="RiskEngine",
    system_message="You are a risk analyst. Assess the given trade for regulatory risk.",
    llm_config={"model": "gpt-4o"},
)

user_proxy = UserProxyAgent(name="User", human_input_mode="NEVER")
user_proxy.initiate_chat(risk_agent, message="Assess trade T-984231")
```

To attach the agent's certificate serial to outbound HTTP calls:

```python
from kakunin.integrations.autogen import KakuninConversableAgent, KakuninHttpxMixin

class MyAgent(KakuninHttpxMixin, KakuninConversableAgent):
    pass

agent = MyAgent(
    kakunin=client,
    agent_id="agt-456",
    cert_serial="3A:F2:...",
    name="RiskEngine",
    llm_config={"model": "gpt-4o"},
)

# Outbound requests carry X-Kakunin-Cert-Serial automatically
resp = await agent.http_client.get("https://internal-service/prices")
```

---

## Other Framework Integrations

| Framework | Import | What it does |
|---|---|---|
| LangGraph | `kakunin.integrations.langgraph.kakunin_node` | Decorator: emits behavioral event on every node execution |
| LlamaIndex | `kakunin.integrations.llamaindex.KakuninFunctionToolGuard` | Wraps FunctionTool with scope check |
| CrewAI | `kakunin.integrations.crewai.KakuninCrewAgent` | Agent subclass with event emission + scope guard |
| CAMEL-AI | `kakunin.integrations.camel.KakuninToolkit` | 4 FunctionTools (verify status, check scope, risk score, emit event) |

---

## Error Handling

```python
from kakunin import ScopeViolationError, RateLimitError, AuthenticationError

try:
    result = await guarded_function()
except ScopeViolationError as e:
    # Agent is suspended, retired, or missing a required scope
    print(f"Scope check failed: {e.missing_scopes}")
    # Do NOT proceed — the agent is not authorised
except RateLimitError:
    await asyncio.sleep(1)
    # Retry
except AuthenticationError:
    # Invalid API key — check KAKUNIN_API_KEY env var
    raise
```

---

Full docs at [kakunin.ai/docs](https://kakunin.ai/docs).
