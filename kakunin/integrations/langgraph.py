"""
LangGraph integration shim.

Wraps any LangGraph node callable so every invocation emits a Kakunin
behavioral event automatically.

Usage::

    from kakunin import Kakunin
    from kakunin.integrations.langgraph import kakunin_node

    client = Kakunin(api_key="kak_live_...")

    # Wrap an existing node function
    @kakunin_node(client, agent_id="agt-123")
    async def risk_engine_node(state: dict) -> dict:
        # your node logic
        return state

    # Or wrap at graph construction time
    graph.add_node("risk_engine", kakunin_node(client, agent_id="agt-123")(risk_engine_node))
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, TypeVar

from ..client import Kakunin

F = TypeVar("F", bound=Callable[..., Any])


def kakunin_node(
    kakunin: Kakunin,
    *,
    agent_id: str,
    action_type: str = "api_call",
    chain_id: str | None = None,
) -> Callable[[F], F]:
    """
    Decorator that wraps a LangGraph node function to emit Kakunin events.

    Emits:
    - action_type (default: api_call) on each node invocation
    - transaction_anomaly if the node raises an exception

    Args:
        kakunin: Authenticated Kakunin client instance.
        agent_id: Kakunin agent ID for this node.
        action_type: Event type to emit on invocation. Default: "api_call".
        chain_id: Optional decision chain ID to group events.

    Example::

        @kakunin_node(client, agent_id="agt-123", action_type="data_access")
        async def data_ingestion_node(state: dict) -> dict:
            ...
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.monotonic()
            try:
                result = await fn(*args, **kwargs)
                elapsed_ms = int((time.monotonic() - start) * 1000)
                _fire_and_forget(kakunin, agent_id, action_type, chain_id, {
                    "node": fn.__name__,
                    "elapsed_ms": elapsed_ms,
                })
                return result
            except Exception as exc:
                _fire_and_forget(kakunin, agent_id, "transaction_anomaly", chain_id, {
                    "node": fn.__name__,
                    "error": type(exc).__name__,
                })
                raise

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.monotonic()
            try:
                result = fn(*args, **kwargs)
                elapsed_ms = int((time.monotonic() - start) * 1000)
                _fire_and_forget(kakunin, agent_id, action_type, chain_id, {
                    "node": fn.__name__,
                    "elapsed_ms": elapsed_ms,
                })
                return result
            except Exception as exc:
                _fire_and_forget(kakunin, agent_id, "transaction_anomaly", chain_id, {
                    "node": fn.__name__,
                    "error": type(exc).__name__,
                })
                raise

        import inspect as _inspect
        wrapper = async_wrapper if _inspect.iscoroutinefunction(fn) else sync_wrapper
        return wrapper  # type: ignore[return-value]

    return decorator


def _fire_and_forget(
    kakunin: Kakunin,
    agent_id: str,
    action_type: str,
    chain_id: str | None,
    details: dict[str, Any],
) -> None:
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(_emit(kakunin, agent_id, action_type, chain_id, details))
        else:
            loop.run_until_complete(_emit(kakunin, agent_id, action_type, chain_id, details))
    except Exception:  # noqa: BLE001
        pass


async def _emit(
    kakunin: Kakunin,
    agent_id: str,
    action_type: str,
    chain_id: str | None,
    details: dict[str, Any],
) -> None:
    try:
        await kakunin.events.create(
            agent_id=agent_id,
            action_type=action_type,
            chain_id=chain_id,
            details=details,
        )
    except Exception:  # noqa: BLE001
        pass
