"""Shared base class for all framework shims."""

from __future__ import annotations

import asyncio
from typing import Any

from ..client import Kakunin


class KakuninAgentBase:
    """
    Mixin that wraps an agentic framework agent to auto-emit Kakunin events.

    Subclasses call _emit(action_type, details) at framework hook points.
    Fire-and-forget — never blocks the agent's execution path.
    """

    def __init__(self, kakunin: Kakunin, agent_id: str, **kwargs: Any) -> None:
        self._kakunin = kakunin
        self._kakunin_agent_id = agent_id
        super().__init__(**kwargs)

    def _emit(
        self,
        action_type: str,
        details: dict[str, Any] | None = None,
        chain_id: str | None = None,
    ) -> None:
        """Fire-and-forget event emission — never raises, never blocks."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self._emit_async(action_type, details, chain_id))
            else:
                loop.run_until_complete(self._emit_async(action_type, details, chain_id))
        except Exception:  # noqa: BLE001
            pass  # monitoring must never affect agent execution

    async def _emit_async(
        self,
        action_type: str,
        details: dict[str, Any] | None = None,
        chain_id: str | None = None,
    ) -> None:
        try:
            await self._kakunin.events.create(
                agent_id=self._kakunin_agent_id,
                action_type=action_type,
                chain_id=chain_id,
                details=details or {},
            )
        except Exception:  # noqa: BLE001
            pass

    @property
    def kakunin_cert_serial(self) -> str | None:
        """
        Returns the certificate serial for this agent, for use in
        X-Kakunin-Cert-Serial headers on outbound requests.

        Cached after first call. Returns None if agent has no active cert.
        """
        return getattr(self, '_kakunin_cert_serial_cache', None)
