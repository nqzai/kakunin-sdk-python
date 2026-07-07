"""Tests for framework integration shims — mocks the framework base classes."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kakunin.integrations.langgraph import kakunin_node, _emit
from kakunin.integrations.scope import verify_agent_scope, _check_scope
from kakunin.integrations.langchain import KakuninToolGuard
from kakunin.integrations.llamaindex import KakuninFunctionToolGuard
from kakunin.exceptions import ScopeViolationError
from kakunin.models import AgentStatus


def make_mock_kakunin() -> MagicMock:
    client = MagicMock()
    client.events = MagicMock()
    client.events.create = AsyncMock()
    return client


# ── LangGraph ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_langgraph_node_emits_on_success() -> None:
    client = make_mock_kakunin()

    @kakunin_node(client, agent_id="agt-1", action_type="data_access")
    async def my_node(state: dict) -> dict:
        return {"result": "ok"}

    result = await my_node({"input": "x"})
    assert result == {"result": "ok"}

    # Give the fire-and-forget task a chance to run
    await asyncio.sleep(0)
    client.events.create.assert_called_once()
    call_kwargs = client.events.create.call_args.kwargs
    assert call_kwargs["action_type"] == "data_access"
    assert call_kwargs["agent_id"] == "agt-1"


@pytest.mark.asyncio
async def test_langgraph_node_emits_anomaly_on_exception() -> None:
    client = make_mock_kakunin()

    @kakunin_node(client, agent_id="agt-1")
    async def failing_node(state: dict) -> dict:
        raise ValueError("boom")

    with pytest.raises(ValueError):
        await failing_node({})

    await asyncio.sleep(0)
    call_kwargs = client.events.create.call_args.kwargs
    assert call_kwargs["action_type"] == "transaction_anomaly"


@pytest.mark.asyncio
async def test_langgraph_node_sync_function() -> None:
    client = make_mock_kakunin()

    @kakunin_node(client, agent_id="agt-1")
    def sync_node(state: dict) -> dict:
        return {"done": True}

    result = sync_node({})
    assert result == {"done": True}


@pytest.mark.asyncio
async def test_langgraph_node_passes_chain_id() -> None:
    client = make_mock_kakunin()

    @kakunin_node(client, agent_id="agt-1", chain_id="chn-xyz")
    async def chained_node(state: dict) -> dict:
        return state

    await chained_node({})
    await asyncio.sleep(0)
    call_kwargs = client.events.create.call_args.kwargs
    assert call_kwargs["chain_id"] == "chn-xyz"


@pytest.mark.asyncio
async def test_emit_never_raises() -> None:
    """_emit swallows all exceptions — must never affect agent execution."""
    client = make_mock_kakunin()
    client.events.create.side_effect = RuntimeError("network error")

    # Should not raise
    await _emit(client, "agt-1", "api_call", None, {})


# ── verify_agent_scope ────────────────────────────────────────────────────────

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
async def test_check_scope_passes_active_agent() -> None:
    client = make_mock_kakunin_with_agent(make_active_agent())
    await _check_scope(client, "agt-1", None)


@pytest.mark.asyncio
async def test_check_scope_raises_for_suspended_agent() -> None:
    client = make_mock_kakunin_with_agent(make_suspended_agent())
    with pytest.raises(ScopeViolationError) as exc_info:
        await _check_scope(client, "agt-1", None)
    assert exc_info.value.agent_status == "suspended"
    assert exc_info.value.agent_id == "agt-1"


@pytest.mark.asyncio
async def test_check_scope_raises_for_missing_scope() -> None:
    client = make_mock_kakunin_with_agent(make_active_agent())
    with pytest.raises(ScopeViolationError) as exc_info:
        await _check_scope(client, "agt-1", ["trade.execute", "admin.halt"])
    assert "admin.halt" in exc_info.value.missing_scopes


@pytest.mark.asyncio
async def test_check_scope_passes_with_matching_scopes() -> None:
    client = make_mock_kakunin_with_agent(make_active_agent())
    await _check_scope(client, "agt-1", ["trade.execute", "data.read"])


@pytest.mark.asyncio
async def test_verify_agent_scope_decorator_async() -> None:
    client = make_mock_kakunin_with_agent(make_active_agent())

    @verify_agent_scope(client, agent_id="agt-1")
    async def my_fn(x: int) -> int:
        return x * 2

    result = await my_fn(5)
    assert result == 10
    client.agents.get.assert_called_once_with("agt-1")


@pytest.mark.asyncio
async def test_verify_agent_scope_decorator_blocks_suspended() -> None:
    client = make_mock_kakunin_with_agent(make_suspended_agent())

    @verify_agent_scope(client, agent_id="agt-1")
    async def my_fn() -> str:
        return "should not run"

    with pytest.raises(ScopeViolationError):
        await my_fn()


# ── KakuninToolGuard (LangChain) ──────────────────────────────────────────────

def make_mock_lc_tool(name: str = "mock_tool") -> MagicMock:
    tool = MagicMock()
    tool.name = name
    tool.description = "A mock tool"
    tool._run = MagicMock(return_value="tool_result")
    tool._arun = AsyncMock(return_value="async_tool_result")
    return tool


@pytest.mark.asyncio
async def test_langchain_tool_guard_passes_active_agent() -> None:
    client = make_mock_kakunin_with_agent(make_active_agent())
    lc_tool = make_mock_lc_tool()
    guard = KakuninToolGuard(kakunin=client, agent_id="agt-1", tool=lc_tool)

    result = await guard._arun("some_input")
    assert result == "async_tool_result"
    client.agents.get.assert_called_once()


@pytest.mark.asyncio
async def test_langchain_tool_guard_blocks_suspended_agent() -> None:
    client = make_mock_kakunin_with_agent(make_suspended_agent())
    lc_tool = make_mock_lc_tool()
    guard = KakuninToolGuard(kakunin=client, agent_id="agt-1", tool=lc_tool)

    with pytest.raises(ScopeViolationError):
        await guard._arun("some_input")
    lc_tool._arun.assert_not_called()


@pytest.mark.asyncio
async def test_langchain_tool_guard_metadata() -> None:
    client = make_mock_kakunin_with_agent(make_active_agent())
    lc_tool = make_mock_lc_tool(name="trade_executor")
    guard = KakuninToolGuard(kakunin=client, agent_id="agt-1", tool=lc_tool)
    assert guard.name == "trade_executor"


# ── KakuninFunctionToolGuard (LlamaIndex) ────────────────────────────────────

@pytest.mark.asyncio
async def test_llamaindex_tool_guard_passes_active_agent() -> None:
    client = make_mock_kakunin_with_agent(make_active_agent())

    def my_fn(x: int) -> int:
        return x + 1

    guard = KakuninFunctionToolGuard(
        kakunin=client,
        agent_id="agt-1",
        fn=my_fn,
        name="my_fn",
        description="adds one",
    )
    result = await guard._arun(5)
    assert result == 6
    client.agents.get.assert_called_once()


@pytest.mark.asyncio
async def test_llamaindex_tool_guard_blocks_suspended_agent() -> None:
    client = make_mock_kakunin_with_agent(make_suspended_agent())
    called = False

    def my_fn() -> str:
        nonlocal called
        called = True
        return "should not run"

    guard = KakuninFunctionToolGuard(kakunin=client, agent_id="agt-1", fn=my_fn)
    with pytest.raises(ScopeViolationError):
        await guard._arun()
    assert not called


@pytest.mark.asyncio
async def test_llamaindex_tool_guard_scope_check() -> None:
    client = make_mock_kakunin_with_agent(make_active_agent())

    def my_fn() -> str:
        return "ok"

    guard = KakuninFunctionToolGuard(
        kakunin=client,
        agent_id="agt-1",
        fn=my_fn,
        required_scopes=["admin.halt"],  # not in active agent's scopes
    )
    with pytest.raises(ScopeViolationError) as exc_info:
        await guard._arun()
    assert "admin.halt" in exc_info.value.missing_scopes
