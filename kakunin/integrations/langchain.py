"""
LangChain integration shim.

Provides KakuninToolGuard — wraps any LangChain BaseTool so every invocation
runs a Kakunin scope verification check before the tool executes.
Requires langchain-core ≥0.1.0.

Usage::

    from langchain_core.tools import BaseTool, tool
    from kakunin import Kakunin
    from kakunin.integrations.langchain import KakuninToolGuard

    client = Kakunin(api_key="kak_live_...")

    @tool
    def trade_executor(order: str) -> str:
        \"\"\"Execute a trade order.\"\"\"
        return execute_trade(order)

    guarded = KakuninToolGuard(
        kakunin=client,
        agent_id="agt-123",
        tool=trade_executor,
        required_scopes=["trade.execute"],
    )

    # Drop into a LangChain agent as a normal tool:
    agent = create_react_agent(llm, tools=[guarded], prompt=prompt)

You can also decorate tool methods directly with the framework-agnostic
verify_agent_scope from kakunin.integrations.scope.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

from .scope import _check_scope
from ..client import Kakunin


class KakuninToolGuard:
    """
    Wraps a LangChain tool with a pre-execution Kakunin scope check.

    Implements the LangChain tool protocol (name, description, _run, _arun)
    without hard-importing langchain at module load time.
    """

    def __init__(
        self,
        kakunin: Kakunin,
        agent_id: str,
        tool: Any,
        *,
        required_scopes: list[str] | None = None,
    ) -> None:
        self._kakunin = kakunin
        self._agent_id = agent_id
        self._tool = tool
        self._required_scopes = required_scopes

        # Surface LangChain tool metadata so this object is usable directly
        self.name: str = getattr(tool, "name", type(tool).__name__)
        self.description: str = getattr(tool, "description", "")

    # ── LangChain BaseTool protocol ──────────────────────────────────────────

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        import asyncio
        import concurrent.futures

        async def _verify_and_run() -> Any:
            await _check_scope(self._kakunin, self._agent_id, self._required_scopes)
            if asyncio.iscoroutinefunction(self._tool._run):  # type: ignore[union-attr]
                return await self._tool._run(*args, **kwargs)  # type: ignore[union-attr]
            return self._tool._run(*args, **kwargs)  # type: ignore[union-attr]

        try:
            loop = asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(asyncio.run, _verify_and_run())
                return fut.result()
        except RuntimeError:
            return asyncio.run(_verify_and_run())

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        await _check_scope(self._kakunin, self._agent_id, self._required_scopes)
        if inspect.iscoroutinefunction(self._tool._arun):  # type: ignore[union-attr]
            return await self._tool._arun(*args, **kwargs)  # type: ignore[union-attr]
        return self._tool._run(*args, **kwargs)  # type: ignore[union-attr]

    # Delegate everything else to the underlying tool
    def __getattr__(self, name: str) -> Any:
        return getattr(self._tool, name)

    def __repr__(self) -> str:
        return f"KakuninToolGuard(agent_id={self._agent_id!r}, tool={self.name!r})"


def langchain_scope_callback(
    kakunin: Kakunin,
    agent_id: str,
    *,
    required_scopes: list[str] | None = None,
) -> Any:
    """
    Returns a LangChain BaseCallbackHandler that verifies agent scope at the
    start of every chain run.

    Use this when you want to guard an entire chain rather than individual tools.

    Requires langchain-core ≥0.1.0.

    Example::

        from langchain_core.runnables import RunnableSequence
        from kakunin.integrations.langchain import langchain_scope_callback

        guard = langchain_scope_callback(client, agent_id="agt-123")
        chain = my_chain.with_config(callbacks=[guard])
    """
    try:
        from langchain_core.callbacks import BaseCallbackHandler  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "langchain-core is not installed. Run: pip install langchain-core"
        ) from e

    import asyncio
    import concurrent.futures

    class _KakuninGuardHandler(BaseCallbackHandler):
        def on_chain_start(
            self,
            serialized: dict[str, Any],
            inputs: dict[str, Any],
            **kwargs: Any,
        ) -> None:
            async def _check() -> None:
                await _check_scope(kakunin, agent_id, required_scopes)

            try:
                loop = asyncio.get_running_loop()
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    pool.submit(asyncio.run, _check()).result()
            except RuntimeError:
                asyncio.run(_check())

    return _KakuninGuardHandler()
