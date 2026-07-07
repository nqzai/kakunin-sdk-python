"""
LlamaIndex integration shim.

Provides KakuninFunctionToolGuard — wraps any LlamaIndex FunctionTool or
callable so every invocation runs a Kakunin scope verification check first.
Requires llama-index-core ≥0.10.0.

Usage::

    from llama_index.core.tools import FunctionTool
    from kakunin import Kakunin
    from kakunin.integrations.llamaindex import KakuninFunctionToolGuard

    client = Kakunin(api_key="kak_live_...")

    def trade_executor(order: str) -> str:
        return execute_trade(order)

    guarded = KakuninFunctionToolGuard(
        kakunin=client,
        agent_id="agt-123",
        fn=trade_executor,
        name="trade_executor",
        description="Execute a trade order",
        required_scopes=["trade.execute"],
    )

    # Pass to a LlamaIndex agent:
    agent = ReActAgent.from_tools([guarded], llm=llm, verbose=True)

You can also use the framework-agnostic verify_agent_scope decorator from
kakunin.integrations.scope for a simpler one-liner approach.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
from typing import Any, Callable

from .scope import _check_scope
from ..client import Kakunin


class KakuninFunctionToolGuard:
    """
    LlamaIndex-compatible tool wrapper with Kakunin scope verification.

    Implements the LlamaIndex AsyncBaseTool / BaseTool protocol so it can be
    dropped into any LlamaIndex agent without modification.
    """

    def __init__(
        self,
        kakunin: Kakunin,
        agent_id: str,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        description: str = "",
        required_scopes: list[str] | None = None,
    ) -> None:
        self._kakunin = kakunin
        self._agent_id = agent_id
        self._fn = fn
        self._required_scopes = required_scopes
        self.metadata = _ToolMetadata(
            name=name or fn.__name__,
            description=description or (fn.__doc__ or "").strip(),
        )

    # ── LlamaIndex tool protocol ─────────────────────────────────────────────

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._run(*args, **kwargs)

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        try:
            loop = asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(asyncio.run, _check_scope(self._kakunin, self._agent_id, self._required_scopes))
                fut.result()
        except RuntimeError:
            asyncio.run(_check_scope(self._kakunin, self._agent_id, self._required_scopes))
        return self._fn(*args, **kwargs)

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        await _check_scope(self._kakunin, self._agent_id, self._required_scopes)
        if inspect.iscoroutinefunction(self._fn):
            return await self._fn(*args, **kwargs)
        return self._fn(*args, **kwargs)

    # LlamaIndex agents call tool.call(input) where input is a ToolInput
    def call(self, input: Any) -> Any:  # noqa: A002
        # LlamaIndex ToolInput has .input (str) or kwargs
        if hasattr(input, "kwargs"):
            return self._run(**input.kwargs)
        raw = getattr(input, "input", input)
        return self._run(raw)

    async def acall(self, input: Any) -> Any:  # noqa: A002
        if hasattr(input, "kwargs"):
            return await self._arun(**input.kwargs)
        raw = getattr(input, "input", input)
        return await self._arun(raw)

    def __repr__(self) -> str:
        return (
            f"KakuninFunctionToolGuard(agent_id={self._agent_id!r}, "
            f"fn={self.metadata.name!r})"
        )


class _ToolMetadata:
    """Minimal stand-in for LlamaIndex ToolMetadata — no hard import needed."""

    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description
        self.fn_schema = None

    def __repr__(self) -> str:
        return f"ToolMetadata(name={self.name!r})"
