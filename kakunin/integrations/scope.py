"""
Framework-agnostic verify_agent_scope decorator.

Performs a pre-flight agent status + scope check before the wrapped function
runs. Raises ScopeViolationError if the agent is suspended, retired, or lacks
a required scope listed in agent.metadata["scopes"].

Usage::

    from kakunin import Kakunin
    from kakunin.integrations.scope import verify_agent_scope

    client = Kakunin(api_key="kak_live_...")

    @verify_agent_scope(client, agent_id="agt-123", required_scopes=["trade.execute"])
    async def execute_trade(order: dict) -> dict:
        ...

    # Also works on sync functions and on any framework's callbacks.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import inspect
from typing import Any, Callable, TypeVar

from ..client import Kakunin
from ..exceptions import ScopeViolationError
from ..models import AgentStatus

F = TypeVar("F", bound=Callable[..., Any])


async def _check_scope(
    kakunin: Kakunin,
    agent_id: str,
    required_scopes: list[str] | None,
) -> None:
    """Fetch agent record and assert it is active with the required scopes."""
    agent = await kakunin.agents.get(agent_id)
    if agent.status != AgentStatus.active:
        raise ScopeViolationError(
            f"Agent {agent_id!r} is {agent.status.value!r} — only active agents may execute guarded operations.",
            agent_id=agent_id,
            agent_status=agent.status.value,
        )
    if required_scopes:
        permitted: list[str] = (agent.metadata or {}).get("scopes", [])
        missing = [s for s in required_scopes if s not in permitted]
        if missing:
            raise ScopeViolationError(
                f"Agent {agent_id!r} missing required scopes: {missing}. "
                f"Permitted: {permitted}",
                agent_id=agent_id,
                agent_status=agent.status.value,
                missing_scopes=missing,
            )


def verify_agent_scope(
    kakunin: Kakunin,
    agent_id: str,
    *,
    required_scopes: list[str] | None = None,
) -> Callable[[F], F]:
    """
    Decorator: verify agent is active (and optionally holds required scopes)
    before the wrapped function executes.

    Raises ScopeViolationError — never swallows it — unlike the event-emission
    helpers. The caller must decide whether to fallback or abort.

    Works with both async and sync functions. For sync functions called from
    within a running event loop the scope check runs in a thread pool so it
    never blocks the loop.

    Args:
        kakunin: Authenticated Kakunin client.
        agent_id: Kakunin agent ID to verify.
        required_scopes: Optional list of scope strings that must appear in
            agent.metadata["scopes"]. Pass None to skip scope check.

    Example::

        @verify_agent_scope(client, agent_id="agt-123", required_scopes=["data.read"])
        async def fetch_portfolio(account_id: str) -> dict:
            ...
    """

    def decorator(fn: F) -> F:
        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                await _check_scope(kakunin, agent_id, required_scopes)
                return await fn(*args, **kwargs)

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                loop = asyncio.get_running_loop()
                # Running loop: delegate to a thread pool to avoid deadlock
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    fut = pool.submit(asyncio.run, _check_scope(kakunin, agent_id, required_scopes))
                    fut.result()  # re-raises ScopeViolationError if check fails
            except RuntimeError:
                # No running event loop — call directly
                asyncio.run(_check_scope(kakunin, agent_id, required_scopes))
            return fn(*args, **kwargs)

        return sync_wrapper  # type: ignore[return-value]

    return decorator  # type: ignore[return-value]
