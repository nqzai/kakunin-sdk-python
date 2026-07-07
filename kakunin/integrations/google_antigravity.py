"""
Google Antigravity SDK integration shim for Kakunin.

Provides lifecycle hooks to monitor agent behavior and enforce tool-level scope compliance.
"""

from __future__ import annotations

from typing import Any, Optional

from .scope import _check_scope
from ..client import Kakunin

try:
    try:
        from google.adk.hooks import (
            OnSessionStartHook,
            OnSessionEndHook,
            PreTurnHook,
            PostTurnHook,
            PreToolCallDecideHook,
            PostToolCallHook,
            OnToolErrorHook,
            HookContext,
        )
        from google.adk.types import HookResult, ToolCall
    except ImportError:
        from google.antigravity.hooks import (
            OnSessionStartHook,
            OnSessionEndHook,
            PreTurnHook,
            PostTurnHook,
            PreToolCallDecideHook,
            PostToolCallHook,
            OnToolErrorHook,
            HookContext,
        )
        from google.antigravity.types import HookResult, ToolCall
    HAS_ANTIGRAVITY = True
except ImportError:
    # Dummy fallbacks to avoid import errors when neither is installed
    class OnSessionStartHook: pass  # type: ignore[no-redef]
    class OnSessionEndHook: pass  # type: ignore[no-redef]
    class PreTurnHook: pass  # type: ignore[no-redef]
    class PostTurnHook: pass  # type: ignore[no-redef]
    class PreToolCallDecideHook: pass  # type: ignore[no-redef]
    class PostToolCallHook: pass  # type: ignore[no-redef]
    class OnToolErrorHook: pass  # type: ignore[no-redef]
    class HookContext: pass  # type: ignore[no-redef]
    class HookResult:  # type: ignore[no-redef]
        def __init__(self, allow: bool, reason: str | None = None) -> None:
            self.allow = allow
            self.reason = reason
    class ToolCall: pass  # type: ignore[no-redef]
    HAS_ANTIGRAVITY = False


class KakuninSessionStartHook(OnSessionStartHook):  # type: ignore[misc]
    """Logs the session start event in Kakunin."""

    def __init__(self, kakunin: Kakunin, agent_id: str, chain_id: str | None = None) -> None:
        self._kakunin = kakunin
        self._agent_id = agent_id
        self._chain_id = chain_id

    async def run(self, context: HookContext, data: Any = None) -> None:
        try:
            await self._kakunin.events.create(
                agent_id=self._agent_id,
                action_type="session_started",
                chain_id=self._chain_id,
            )
        except Exception:
            pass


class KakuninSessionEndHook(OnSessionEndHook):  # type: ignore[misc]
    """Logs the session end event in Kakunin."""

    def __init__(self, kakunin: Kakunin, agent_id: str, chain_id: str | None = None) -> None:
        self._kakunin = kakunin
        self._agent_id = agent_id
        self._chain_id = chain_id

    async def run(self, context: HookContext, data: Any = None) -> None:
        try:
            await self._kakunin.events.create(
                agent_id=self._agent_id,
                action_type="session_ended",
                chain_id=self._chain_id,
            )
        except Exception:
            pass


class KakuninPreTurnHook(PreTurnHook):  # type: ignore[misc]
    """Intercepts and logs the user prompt event in Kakunin before agent execution."""

    def __init__(self, kakunin: Kakunin, agent_id: str, chain_id: str | None = None) -> None:
        self._kakunin = kakunin
        self._agent_id = agent_id
        self._chain_id = chain_id

    async def run(self, context: HookContext, data: str) -> HookResult:
        try:
            await self._kakunin.events.create(
                agent_id=self._agent_id,
                action_type="prompt_received",
                chain_id=self._chain_id,
                details={"prompt": data},
            )
        except Exception:
            pass
        return HookResult(allow=True)


class KakuninPostTurnHook(PostTurnHook):  # type: ignore[misc]
    """Logs the generated assistant response in Kakunin after execution ends."""

    def __init__(self, kakunin: Kakunin, agent_id: str, chain_id: str | None = None) -> None:
        self._kakunin = kakunin
        self._agent_id = agent_id
        self._chain_id = chain_id

    async def run(self, context: HookContext, data: str) -> None:
        try:
            await self._kakunin.events.create(
                agent_id=self._agent_id,
                action_type="response_generated",
                chain_id=self._chain_id,
                details={"response": data},
            )
        except Exception:
            pass


class KakuninPreToolCallDecideHook(PreToolCallDecideHook):  # type: ignore[misc]
    """Verifies that the agent is active and holds the required scopes for a tool."""

    def __init__(
        self,
        kakunin: Kakunin,
        agent_id: str,
        tool_scopes_mapping: dict[str, list[str]],
    ) -> None:
        self._kakunin = kakunin
        self._agent_id = agent_id
        self._tool_scopes_mapping = tool_scopes_mapping

    async def run(self, context: HookContext, data: ToolCall) -> HookResult:
        tool_name = getattr(data, "name", None)
        if not tool_name:
            return HookResult(allow=True)

        required_scopes = self._tool_scopes_mapping.get(tool_name)
        try:
            await _check_scope(self._kakunin, self._agent_id, required_scopes)
        except Exception as e:
            return HookResult(allow=False, reason=str(e))
        return HookResult(allow=True)


class KakuninPostToolCallHook(PostToolCallHook):  # type: ignore[misc]
    """Logs the tool execution event in Kakunin."""

    def __init__(self, kakunin: Kakunin, agent_id: str, chain_id: str | None = None) -> None:
        self._kakunin = kakunin
        self._agent_id = agent_id
        self._chain_id = chain_id

    async def run(self, context: HookContext, data: Any) -> None:
        try:
            tool_name = getattr(context, "tool_name", None) or getattr(data, "name", None)
            await self._kakunin.events.create(
                agent_id=self._agent_id,
                action_type="tool_executed",
                chain_id=self._chain_id,
                details={"tool_name": tool_name} if tool_name else {},
            )
        except Exception:
            pass


class KakuninOnToolErrorHook(OnToolErrorHook):  # type: ignore[misc]
    """Logs tool errors as execution anomalies in Kakunin."""

    def __init__(self, kakunin: Kakunin, agent_id: str, chain_id: str | None = None) -> None:
        self._kakunin = kakunin
        self._agent_id = agent_id
        self._chain_id = chain_id

    async def run(self, context: HookContext, data: Any) -> Optional[str]:
        try:
            tool_name = getattr(context, "tool_name", None)
            await self._kakunin.events.create(
                agent_id=self._agent_id,
                action_type="tool_error",
                chain_id=self._chain_id,
                details={"error": str(data), "tool_name": tool_name} if tool_name else {"error": str(data)},
            )
        except Exception:
            pass
        return None


def get_kakunin_hooks(
    kakunin: Kakunin,
    agent_id: str,
    tool_scopes_mapping: dict[str, list[str]] | None = None,
    chain_id: str | None = None,
) -> list[Any]:
    """
    Convenience function to instantiate and return a list of all Kakunin hooks
    to register on the Google Antigravity LocalAgentConfig.
    """
    hooks = [
        KakuninSessionStartHook(kakunin, agent_id, chain_id),
        KakuninSessionEndHook(kakunin, agent_id, chain_id),
        KakuninPreTurnHook(kakunin, agent_id, chain_id),
        KakuninPostTurnHook(kakunin, agent_id, chain_id),
        KakuninPostToolCallHook(kakunin, agent_id, chain_id),
        KakuninOnToolErrorHook(kakunin, agent_id, chain_id),
    ]
    if tool_scopes_mapping:
        hooks.append(KakuninPreToolCallDecideHook(kakunin, agent_id, tool_scopes_mapping))
    return hooks
