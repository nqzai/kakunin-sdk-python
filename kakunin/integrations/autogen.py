"""
AutoGen integration shim.

Wraps a pyautogen ConversableAgent so every message send/receive emits a
Kakunin behavioral event. Requires pyautogen ≥0.2.0.

Usage::

    from autogen import UserProxyAgent
    from kakunin import Kakunin
    from kakunin.integrations.autogen import KakuninConversableAgent

    client = Kakunin(api_key="kak_live_...")

    risk_agent = KakuninConversableAgent(
        kakunin=client,
        agent_id="agt-456",
        # Standard AutoGen kwargs:
        name="RiskEngine",
        system_message="You are a risk analyst...",
        llm_config={"model": "gpt-4o"},
    )

    # Use in multi-agent conversations as normal
    user_proxy = UserProxyAgent(name="User")
    user_proxy.initiate_chat(risk_agent, message="Assess trade T-984231")
"""

from __future__ import annotations

from typing import Any

from ._base import KakuninAgentBase
from .scope import _check_scope


class KakuninConversableAgent(KakuninAgentBase):
    """
    AutoGen ConversableAgent subclass with automatic Kakunin event emission
    and optional pre-reply scope verification.

    Hooks into generate_reply() and receive() to emit events for every
    message in the multi-agent conversation.

    Pass required_scopes to block reply generation if the agent lacks the
    listed scope strings in its Kakunin metadata.

    Example::

        agent = KakuninConversableAgent(
            kakunin=client,
            agent_id="agt-456",
            required_scopes=["chat.reply"],
            name="RiskEngine",
            system_message="...",
            llm_config={"model": "gpt-4o"},
        )
    """

    def __init__(self, kakunin: Any, agent_id: str, required_scopes: list[str] | None = None, **kwargs: Any) -> None:
        self._required_scopes = required_scopes
        try:
            from autogen import ConversableAgent  # type: ignore[import-not-found]
        except ImportError:
            try:
                from pyautogen import ConversableAgent  # type: ignore[import-not-found]
            except ImportError as e:
                raise ImportError(
                    "pyautogen is not installed. Run: pip install pyautogen"
                ) from e

        self.__class__ = type(
            "KakuninConversableAgent",
            (KakuninAgentBase, ConversableAgent),
            dict(KakuninConversableAgent.__dict__),
        )
        super().__init__(kakunin=kakunin, agent_id=agent_id, **kwargs)

    def _verify_scope_sync(self) -> None:
        """Synchronous scope check — runs before generate_reply if required_scopes set."""
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

    def receive(
        self,
        message: dict[str, Any] | str,
        sender: Any,
        request_reply: bool | None = None,
        silent: bool | None = False,
    ) -> None:
        content = message.get("content", str(message)) if isinstance(message, dict) else message
        self._emit("data_access", {
            "direction": "receive",
            "sender": getattr(sender, "name", str(sender)),
            "content_length": len(str(content)),
        })
        super().receive(message, sender, request_reply=request_reply, silent=silent)  # type: ignore[misc]

    def generate_reply(
        self,
        messages: list[dict[str, Any]] | None = None,
        sender: Any = None,
        **kwargs: Any,
    ) -> str | dict[str, Any] | None:
        self._verify_scope_sync()
        self._emit("api_call", {
            "direction": "send",
            "message_count": len(messages) if messages else 0,
            "sender": getattr(sender, "name", str(sender)) if sender else None,
        })
        try:
            reply: str | dict[str, Any] | None = super().generate_reply(messages=messages, sender=sender, **kwargs)  # type: ignore[misc]
            return reply
        except Exception as exc:
            self._emit("transaction_anomaly", {"error": type(exc).__name__})
            raise


class KakuninHttpxMixin:
    """
    Mixin for attaching X-Kakunin-Cert-Serial to outbound httpx requests.

    Use when your AutoGen agent makes HTTP calls via httpx and you want
    downstream services to verify the agent's identity.

    Usage::

        import httpx
        from kakunin.integrations.autogen import KakuninHttpxMixin

        class MyAgent(KakuninHttpxMixin, KakuninConversableAgent):
            pass

        agent = MyAgent(kakunin=client, agent_id="agt-123", cert_serial="3A:F2:...", ...)

        # Use agent.http_client for outbound calls — serial attached automatically
        resp = await agent.http_client.get("https://internal-service/prices")
    """

    def __init__(self, *args: Any, cert_serial: str | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._cert_serial = cert_serial

    @property
    def http_client(self) -> Any:
        try:
            import httpx  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError("httpx is not installed. Run: pip install httpx") from e

        headers = {}
        if self._cert_serial:
            headers["X-Kakunin-Cert-Serial"] = self._cert_serial
        return httpx.AsyncClient(headers=headers)
