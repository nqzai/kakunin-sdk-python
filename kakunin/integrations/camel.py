"""
CAMEL-AI integration shim.

Provides two classes:

1. **KakuninToolkit** — wraps Kakunin's API as CAMEL FunctionTools so any
   CAMEL ChatAgent can verify certificates, check scopes, read risk scores,
   and emit behavioral events. Requires ``camel-ai>=0.2.0``.

2. **KakuninCamelAgent** — subclasses CAMEL ChatAgent with automatic
   Kakunin behavioral event emission on every agent step. Also supports
   optional pre-step scope verification via ``required_scopes``.

Usage (toolkit)::

    from camel.agents import ChatAgent
    from camel.models import ModelFactory
    from kakunin import Kakunin
    from kakunin.integrations.camel import KakuninToolkit

    client = Kakunin(api_key="kak_live_...")

    toolkit = KakuninToolkit(kakunin=client)
    model = ModelFactory.create(model_platform=..., model_type=...)
    agent = ChatAgent(model=model, tools=toolkit.get_tools())

Usage (monitored agent)::

    from kakunin.integrations.camel import KakuninCamelAgent

    agent = KakuninCamelAgent(
        kakunin=client,
        agent_id="agt-123",
        required_scopes=["trade.execute"],   # optional
        model=model,
        system_message=sys_msg,
    )
    response = agent.step(user_message)
"""

from __future__ import annotations

import asyncio
from typing import Any

from ._base import KakuninAgentBase
from .scope import _check_scope
from ..client import Kakunin


# ---------------------------------------------------------------------------
# KakuninToolkit
# ---------------------------------------------------------------------------


class KakuninToolkit:
    """
    CAMEL BaseToolkit that exposes Kakunin's compliance API as FunctionTools.

    Wraps four Kakunin operations:
    - verify_agent_certificate
    - check_agent_scope
    - get_risk_score
    - emit_behavior_event

    Requires ``camel-ai>=0.2.0``. Install via:
        pip install kakunin[camel]
    """

    def __init__(self, kakunin: Kakunin) -> None:
        self._kakunin = kakunin

    # ── tool implementations ────────────────────────────────────────────────

    async def verify_agent_certificate(self, agent_id: str) -> dict[str, Any]:
        """
        Verify the compliance status of an AI agent before trusting its claims.

        Returns the agent's current status and the scopes recorded in its
        Kakunin metadata. Only ``active`` agents should be trusted to act.

        Args:
            agent_id: Kakunin agent ID (e.g. ``agt-abc123``) or DID.

        Returns:
            dict with keys: agent_id, agent_status, active, scopes.
        """
        agent = await self._kakunin.agents.get(agent_id)
        return {
            "agent_id": agent_id,
            "agent_status": agent.status.value,
            "active": agent.status.value == "active",
            "scopes": (agent.metadata or {}).get("scopes", []),
        }

    async def check_agent_scope(self, agent_id: str, action: str) -> dict[str, Any]:
        """
        Check whether a specific action is within an agent's permitted scope.

        Use this before allowing an agent to execute a privileged operation
        (e.g. financial_trade, data_mutation). Returns allowed=True/False plus
        the full permitted scope list.

        Args:
            agent_id: Kakunin agent ID.
            action: Action string to check (e.g. ``trade.execute``, ``data.write``).

        Returns:
            dict with keys: allowed (bool), action, permitted_scopes.
        """
        agent = await self._kakunin.agents.get(agent_id)
        permitted: list[str] = (agent.metadata or {}).get("scopes", [])
        return {
            "agent_id": agent_id,
            "action": action,
            "allowed": action in permitted,
            "permitted_scopes": permitted,
            "agent_status": agent.status.value,
        }

    async def get_risk_score(self, agent_id: str) -> dict[str, Any]:
        """
        Return the current behavioral risk score for an AI agent.

        Risk bands: low (<0.3), medium (>=0.3), high (>=0.75),
        critical (>=0.85 — auto-revocation threshold).

        Args:
            agent_id: Kakunin agent ID.

        Returns:
            dict with keys: score (0.0–1.0), band, agent_id.
        """
        agent = await self._kakunin.agents.get(agent_id)
        score: float = (agent.metadata or {}).get("risk_score", 0.0)
        if score >= 0.85:
            band = "critical"
        elif score >= 0.75:
            band = "high"
        elif score >= 0.3:
            band = "medium"
        else:
            band = "low"
        return {"agent_id": agent_id, "score": score, "band": band}

    async def emit_behavior_event(
        self,
        agent_id: str,
        action_type: str,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Emit a behavioral event for an AI agent to Kakunin's audit trail.

        Events are used for real-time risk scoring and EU AI Act Article 12
        logging. This is a fire-and-forget call — it does not block.

        Valid action_type values: api_call, authentication_attempt,
        authentication_failure, data_access, data_mutation,
        transaction_initiated, transaction_anomaly,
        unauthorized_access_attempt, message_signed,
        message_verification_failed.

        Args:
            agent_id: Kakunin agent ID.
            action_type: One of the canonical Kakunin event types.
            details: Optional dict of extra context to store with the event.

        Returns:
            dict with keys: event_id, agent_id, action_type.
        """
        event = await self._kakunin.events.create(
            agent_id=agent_id,
            action_type=action_type,
            details=details or {},
        )
        return {
            "event_id": event.id,
            "agent_id": agent_id,
            "action_type": action_type,
        }

    # ── CAMEL BaseToolkit protocol ──────────────────────────────────────────

    def get_tools(self) -> list[Any]:
        """
        Return CAMEL FunctionTool objects for all four Kakunin operations.

        Each FunctionTool wraps an async method — CAMEL's executor handles
        the event loop. Register the returned list in your ChatAgent's
        ``tools`` parameter.

        Raises:
            ImportError: If ``camel-ai`` is not installed.
        """
        try:
            from camel.toolkits import FunctionTool  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "camel-ai is not installed. Run: pip install kakunin[camel]"
            ) from exc

        return [
            FunctionTool(self.verify_agent_certificate),
            FunctionTool(self.check_agent_scope),
            FunctionTool(self.get_risk_score),
            FunctionTool(self.emit_behavior_event),
        ]


# ---------------------------------------------------------------------------
# KakuninCamelAgent
# ---------------------------------------------------------------------------


class KakuninCamelAgent(KakuninAgentBase):
    """
    CAMEL ChatAgent subclass with automatic Kakunin behavioral event emission.

    Every call to ``step()`` emits an ``api_call`` event to Kakunin after the
    agent responds. Emission is fire-and-forget — it never blocks or raises.

    Optionally pass ``required_scopes`` to block step execution when the agent
    lacks the listed scope strings in its Kakunin certificate metadata.

    Requires ``camel-ai>=0.2.0``. Install via:
        pip install kakunin[camel]

    Example::

        from camel.agents import ChatAgent
        from camel.messages import BaseMessage
        from kakunin import Kakunin
        from kakunin.integrations.camel import KakuninCamelAgent

        client = Kakunin(api_key="kak_live_...")

        agent = KakuninCamelAgent(
            kakunin=client,
            agent_id="agt-123",
            required_scopes=["trade.execute"],
            model=model,
            system_message=sys_msg,
        )

        user_msg = BaseMessage.make_user_message(role_name="User", content="Assess risk")
        response = agent.step(user_msg)
    """

    def __init__(
        self,
        kakunin: Kakunin,
        agent_id: str,
        required_scopes: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            from camel.agents import ChatAgent  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "camel-ai is not installed. Run: pip install kakunin[camel]"
            ) from exc

        # KakuninAgentBase must come first so MRO resolves _emit correctly
        # ChatAgent.__init__ takes model + system_message kwargs
        KakuninAgentBase.__init__(self, kakunin=kakunin, agent_id=agent_id)
        ChatAgent.__init__(self, **kwargs)

        self._required_scopes = required_scopes
        self.__class__.__bases__ = (KakuninAgentBase, ChatAgent)

    def step(self, *args: Any, **kwargs: Any) -> Any:
        """Run one agent step, optionally scope-checking first, then emit event."""
        if self._required_scopes:
            # Synchronous scope check before delegating to ChatAgent
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        pool.submit(
                            asyncio.run,
                            _check_scope(self._kakunin, self._kakunin_agent_id, self._required_scopes),
                        ).result()
                else:
                    loop.run_until_complete(
                        _check_scope(self._kakunin, self._kakunin_agent_id, self._required_scopes)
                    )
            except Exception:
                raise  # ScopeViolationError propagates; monitoring errors swallowed below

        # Resolve ChatAgent via MRO — don't hardcode super() to avoid MRO issues
        import camel.agents  # type: ignore[import-not-found]
        result = camel.agents.ChatAgent.step(self, *args, **kwargs)

        self._emit("api_call", {"framework": "camel-ai"})
        return result

    async def astep(self, *args: Any, **kwargs: Any) -> Any:
        """Async step with scope check and event emission."""
        if self._required_scopes:
            await _check_scope(self._kakunin, self._kakunin_agent_id, self._required_scopes)

        import camel.agents  # type: ignore[import-not-found]
        result = await camel.agents.ChatAgent.astep(self, *args, **kwargs)  # type: ignore[attr-defined]

        self._emit("api_call", {"framework": "camel-ai"})
        return result
