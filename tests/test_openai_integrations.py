"""Tests for OpenAI Swarm and Assistants API integrations."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kakunin.integrations.openai_swarm import KakuninSwarm
from kakunin.integrations.openai_assistants import handle_assistants_requires_action
from kakunin.exceptions import ScopeViolationError
from kakunin.models import AgentStatus


def make_mock_kakunin() -> MagicMock:
    client = MagicMock()
    client.events = MagicMock()
    client.events.create = AsyncMock()
    return client


def make_active_agent() -> MagicMock:
    agent = MagicMock()
    agent.status = AgentStatus.active
    agent.metadata = {"scopes": ["trade.execute"]}
    return agent


def make_suspended_agent() -> MagicMock:
    agent = MagicMock()
    agent.status = AgentStatus.suspended
    agent.metadata = {}
    return agent


def make_mock_kakunin_with_agent(agent: MagicMock) -> MagicMock:
    client = make_mock_kakunin()
    client.agents = MagicMock()
    client.agents.get = AsyncMock(return_value=agent)
    return client


class DummyAgent:
    def __init__(self, functions: list) -> None:
        self.functions = functions


# ── OpenAI Swarm Tests ────────────────────────────────────────────────────────

class DummyResponse:
    def __init__(self, content: str) -> None:
        self.messages = [{"role": "assistant", "content": content}]


@pytest.mark.asyncio
async def test_swarm_runs_successfully() -> None:
    client = make_mock_kakunin_with_agent(make_active_agent())

    wrapped_fns = []

    def mock_run(agent, messages, **kwargs):
        # Capture the wrapped functions during execution, before finally block restores them
        nonlocal wrapped_fns
        wrapped_fns = list(agent.functions)
        return DummyResponse("Done!")

    # Patch local Swarm import in the integration with create=True
    with patch("kakunin.integrations.openai_swarm.Swarm.run", side_effect=mock_run, create=True):
        swarm = KakuninSwarm(client, "agt-123")

        async def test_tool() -> str:
            return "tool success"

        agent = DummyAgent([test_tool])

        res = swarm.run(
            agent=agent,
            messages=[{"role": "user", "content": "hello"}],
            tool_scopes_mapping={"test_tool": ["trade.execute"]},
        )
        assert res.messages[-1]["content"] == "Done!"
        
        # Test function wrapping using captured function
        assert len(wrapped_fns) == 1
        wrapped_fn = wrapped_fns[0]
        
        tool_res = await wrapped_fn()
        assert tool_res == "tool success"
        
        # Check active agent query
        client.agents.get.assert_called_with("agt-123")


@pytest.mark.asyncio
async def test_swarm_tool_raises_scope_error() -> None:
    client = make_mock_kakunin_with_agent(make_suspended_agent())

    wrapped_fns = []

    def mock_run(agent, messages, **kwargs):
        nonlocal wrapped_fns
        wrapped_fns = list(agent.functions)
        return DummyResponse("Error")

    with patch("kakunin.integrations.openai_swarm.Swarm.run", side_effect=mock_run, create=True):
        swarm = KakuninSwarm(client, "agt-123")

        async def test_tool() -> str:
            return "success"

        agent = DummyAgent([test_tool])

        swarm.run(
            agent=agent,
            messages=[{"role": "user", "content": "run trade"}],
            tool_scopes_mapping={"test_tool": ["trade.execute"]},
        )

        assert len(wrapped_fns) == 1
        wrapped_fn = wrapped_fns[0]
        
        with pytest.raises(ScopeViolationError):
            await wrapped_fn()

        await asyncio.sleep(0)
        client.events.create.assert_any_call(
            agent_id="agt-123",
            action_type="unauthorized_access_attempt",
            details={"tool_name": "test_tool", "error": "Agent 'agt-123' is 'suspended' — only active agents may execute guarded operations."},
        )


# ── OpenAI Assistants API Tests ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_assistants_executes_successfully() -> None:
    agent = make_active_agent()
    client = make_mock_kakunin_with_agent(agent)

    # Set up mock run structure properly
    func_mock = MagicMock()
    func_mock.name = "my_tool"
    func_mock.arguments = '{"param": "val"}'

    tool_call = MagicMock()
    tool_call.id = "call_abc"
    tool_call.function = func_mock

    run = MagicMock()
    run.required_action.submit_tool_outputs.tool_calls = [tool_call]

    called_args = {}

    async def my_tool(param: str) -> str:
        called_args["param"] = param
        return "tool_response"

    outputs = await handle_assistants_requires_action(
        kakunin=client,
        agent_id="agt-123",
        run=run,
        tool_scopes_mapping={"my_tool": ["trade.execute"]},
        tool_funcs={"my_tool": my_tool},
    )

    assert len(outputs) == 1
    assert outputs[0]["tool_call_id"] == "call_abc"
    assert outputs[0]["output"] == "tool_response"
    assert called_args["param"] == "val"
    
    await asyncio.sleep(0)
    client.events.create.assert_called_with(
        agent_id="agt-123",
        action_type="tool_executed",
        details={"tool_name": "my_tool"},
    )


@pytest.mark.asyncio
async def test_assistants_blocks_on_violation() -> None:
    agent = make_suspended_agent()
    client = make_mock_kakunin_with_agent(agent)

    func_mock = MagicMock()
    func_mock.name = "secure_tool"
    func_mock.arguments = "{}"

    tool_call = MagicMock()
    tool_call.id = "call_xyz"
    tool_call.function = func_mock

    run = MagicMock()
    run.required_action.submit_tool_outputs.tool_calls = [tool_call]

    outputs = await handle_assistants_requires_action(
        kakunin=client,
        agent_id="agt-123",
        run=run,
        tool_scopes_mapping={"secure_tool": ["trade.execute"]},
        tool_funcs={"secure_tool": lambda: "ok"},
        catch_exceptions=True,
    )

    assert len(outputs) == 1
    assert outputs[0]["tool_call_id"] == "call_xyz"
    assert "Scope violation check failed" in outputs[0]["output"]
    
    await asyncio.sleep(0)
    client.events.create.assert_called_with(
        agent_id="agt-123",
        action_type="unauthorized_access_attempt",
        details={"tool_name": "secure_tool", "error": "Agent 'agt-123' is 'suspended' — only active agents may execute guarded operations."},
    )
