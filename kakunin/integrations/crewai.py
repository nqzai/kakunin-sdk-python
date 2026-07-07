"""
CrewAI integration shim.

Wraps a CrewAI Agent so every task execution and tool call emits a Kakunin
behavioral event. Requires CrewAI ≥0.28.0.

Usage::

    from crewai import Agent, Task, Crew
    from kakunin import Kakunin
    from kakunin.integrations.crewai import KakuninCrewAgent

    client = Kakunin(api_key="kak_live_...")

    agent = KakuninCrewAgent(
        kakunin=client,
        agent_id="agt-123",
        # Standard CrewAI Agent kwargs:
        role="Risk Analyst",
        goal="Assess trade risk",
        backstory="...",
        tools=[...],
    )

    # Use agent in a Crew as normal — events are emitted automatically
    task = Task(description="Assess trade T-984231", agent=agent)
    crew = Crew(agents=[agent], tasks=[task])
    crew.kickoff()
"""

from __future__ import annotations

from typing import Any

from ._base import KakuninAgentBase
from .scope import _check_scope


class KakuninCrewAgent(KakuninAgentBase):
    """
    CrewAI Agent subclass with automatic Kakunin behavioral event emission
    and optional pre-execution scope verification.

    Hooks into execute_task() and _use_tool() to emit events without
    requiring any changes to your task or crew configuration.

    Pass required_scopes to block task execution if the agent lacks the
    listed scope strings in its Kakunin metadata.

    Example::

        agent = KakuninCrewAgent(
            kakunin=client,
            agent_id="agt-123",
            required_scopes=["trade.execute"],
            role="Risk Analyst",
            goal="...",
            backstory="...",
        )
    """

    def __init__(self, kakunin: Any, agent_id: str, required_scopes: list[str] | None = None, **kwargs: Any) -> None:
        self._required_scopes = required_scopes
        try:
            from crewai import Agent  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "crewai is not installed. Run: pip install crewai"
            ) from e

        # Patch MRO: KakuninAgentBase → crewai.Agent
        self.__class__ = type(
            "KakuninCrewAgent",
            (KakuninAgentBase, Agent),
            dict(KakuninCrewAgent.__dict__),
        )
        super().__init__(kakunin=kakunin, agent_id=agent_id, **kwargs)

    def _verify_scope_sync(self) -> None:
        """Synchronous scope check — runs before execute_task if required_scopes set."""
        import asyncio
        import concurrent.futures

        if not self._required_scopes:
            return
        try:
            loop = asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(
                    asyncio.run,
                    _check_scope(self._kakunin, self._kakunin_agent_id, self._required_scopes),
                ).result()
        except RuntimeError:
            asyncio.run(_check_scope(self._kakunin, self._kakunin_agent_id, self._required_scopes))

    def execute_task(self, task: Any, context: Any = None, tools: Any = None) -> Any:
        self._verify_scope_sync()
        self._emit("api_call", {"task": str(task)[:200]})
        try:
            result = super().execute_task(task, context=context, tools=tools)  # type: ignore[misc]
            self._emit("data_access", {"task": str(task)[:200], "status": "completed"})
            return result
        except Exception as exc:
            self._emit("transaction_anomaly", {"task": str(task)[:200], "error": type(exc).__name__})
            raise

    def _use_tool(self, tool: Any, input_data: Any) -> Any:
        self._emit("data_access", {"tool": getattr(tool, "name", str(tool)), "input": str(input_data)[:200]})
        return super()._use_tool(tool, input_data)  # type: ignore[misc]
