"""Tests for Google Antigravity SDK integration."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from kakunin.integrations.google_antigravity import (
    KakuninSessionStartHook,
    KakuninSessionEndHook,
    KakuninPreTurnHook,
    KakuninPostTurnHook,
    KakuninPreToolCallDecideHook,
    KakuninPostToolCallHook,
    KakuninOnToolErrorHook,
    get_kakunin_hooks,
)
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
    agent.metadata = {"scopes": ["trade.execute", "data.read"]}
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


@pytest.mark.asyncio
async def test_session_hooks() -> None:
    client = make_mock_kakunin()
    
    start_hook = KakuninSessionStartHook(client, "agt-123", "chn-456")
    await start_hook.run(None)
    client.events.create.assert_called_once_with(
        agent_id="agt-123",
        action_type="session_started",
        chain_id="chn-456",
    )
    
    client.events.create.reset_mock()
    
    end_hook = KakuninSessionEndHook(client, "agt-123", "chn-456")
    await end_hook.run(None)
    client.events.create.assert_called_once_with(
        agent_id="agt-123",
        action_type="session_ended",
        chain_id="chn-456",
    )


@pytest.mark.asyncio
async def test_turn_hooks() -> None:
    client = make_mock_kakunin()
    
    pre_hook = KakuninPreTurnHook(client, "agt-123", "chn-456")
    res = await pre_hook.run(None, "Hello assistant")
    assert res.allow is True
    client.events.create.assert_called_once_with(
        agent_id="agt-123",
        action_type="prompt_received",
        chain_id="chn-456",
        details={"prompt": "Hello assistant"},
    )
    
    client.events.create.reset_mock()
    
    post_hook = KakuninPostTurnHook(client, "agt-123", "chn-456")
    await post_hook.run(None, "Hello human")
    client.events.create.assert_called_once_with(
        agent_id="agt-123",
        action_type="response_generated",
        chain_id="chn-456",
        details={"response": "Hello human"},
    )


@pytest.mark.asyncio
async def test_pre_tool_call_decide_hook_passes() -> None:
    agent = make_active_agent()
    client = make_mock_kakunin_with_agent(agent)
    
    hook = KakuninPreToolCallDecideHook(
        client,
        "agt-123",
        {"trade_tool": ["trade.execute"]}
    )
    
    tool_call = MagicMock()
    tool_call.name = "trade_tool"
    
    res = await hook.run(None, tool_call)
    assert res.allow is True


@pytest.mark.asyncio
async def test_pre_tool_call_decide_hook_blocks_suspended() -> None:
    agent = make_suspended_agent()
    client = make_mock_kakunin_with_agent(agent)
    
    hook = KakuninPreToolCallDecideHook(
        client,
        "agt-123",
        {"trade_tool": ["trade.execute"]}
    )
    
    tool_call = MagicMock()
    tool_call.name = "trade_tool"
    
    res = await hook.run(None, tool_call)
    assert res.allow is False
    assert "suspended" in res.reason


@pytest.mark.asyncio
async def test_pre_tool_call_decide_hook_blocks_suspended_unmapped_tool() -> None:
    agent = make_suspended_agent()
    client = make_mock_kakunin_with_agent(agent)
    
    # Empty mapping
    hook = KakuninPreToolCallDecideHook(
        client,
        "agt-123",
        {}
    )
    
    tool_call = MagicMock()
    tool_call.name = "unmapped_tool"
    
    res = await hook.run(None, tool_call)
    assert res.allow is False
    assert "suspended" in res.reason


@pytest.mark.asyncio
async def test_pre_tool_call_decide_hook_blocks_missing_scopes() -> None:
    agent = make_active_agent()
    client = make_mock_kakunin_with_agent(agent)
    
    hook = KakuninPreToolCallDecideHook(
        client,
        "agt-123",
        {"admin_tool": ["admin.halt"]}
    )
    
    tool_call = MagicMock()
    tool_call.name = "admin_tool"
    
    res = await hook.run(None, tool_call)
    assert res.allow is False
    assert "missing required scopes" in res.reason


@pytest.mark.asyncio
async def test_post_tool_call_hook() -> None:
    client = make_mock_kakunin()
    hook = KakuninPostToolCallHook(client, "agt-123", "chn-456")
    
    context = MagicMock()
    context.tool_name = "test_tool"
    
    await hook.run(context, None)
    client.events.create.assert_called_once_with(
        agent_id="agt-123",
        action_type="tool_executed",
        chain_id="chn-456",
        details={"tool_name": "test_tool"},
    )


@pytest.mark.asyncio
async def test_on_tool_error_hook() -> None:
    client = make_mock_kakunin()
    hook = KakuninOnToolErrorHook(client, "agt-123", "chn-456")
    
    context = MagicMock()
    context.tool_name = "failing_tool"
    
    res = await hook.run(context, ValueError("something went wrong"))
    assert res is None
    client.events.create.assert_called_once_with(
        agent_id="agt-123",
        action_type="tool_error",
        chain_id="chn-456",
        details={"error": "something went wrong", "tool_name": "failing_tool"},
    )


def test_get_kakunin_hooks_factory() -> None:
    client = make_mock_kakunin()
    hooks = get_kakunin_hooks(client, "agt-123", {"tool_a": ["scope_a"]}, "chn-456")
    assert len(hooks) == 7
    assert any(isinstance(h, KakuninPreToolCallDecideHook) for h in hooks)
