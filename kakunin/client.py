"""Kakunin async client — entry point for all API operations."""

from __future__ import annotations

from typing import Any

from ._http import DEFAULT_BASE_URL, DEFAULT_TIMEOUT, HttpTransport
from .models import (
    Agent,
    AgentRisk,
    BehaviorEvent,
    Certificate,
    ChainVerifyResult,
    ComplianceReport,
    DecisionChain,
    HaltReceipt,
    MessageVerifyResult,
    Pagination,
    RiskBand,
    SignResult,
    StandardsFramework,
)


class Kakunin:
    """
    Async client for the Kakunin AI agent compliance API.

    Usage::

        async with Kakunin(api_key="kak_live_...") as client:
            agent = await client.agents.create(
                name="TradeBot-1",
                model="gpt-4o",
                version="2024-11",
                model_hash="sha256:abc123...",
            )
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._http = HttpTransport(api_key=api_key, base_url=base_url, timeout=timeout)
        self.agents = _AgentsResource(self._http)
        self.events = _EventsResource(self._http)
        self.chains = _ChainsResource(self._http)
        self.reports = _ReportsResource(self._http)
        self.verify = _VerifyResource(self._http)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "Kakunin":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()


# ── Resource namespaces ──────────────────────────────────────────────────────

class _AgentsResource:
    def __init__(self, http: HttpTransport) -> None:
        self._http = http

    async def create(
        self,
        *,
        name: str,
        model: str,
        version: str,
        model_hash: str,
        financial_scope: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Agent:
        body: dict[str, Any] = {
            "name": name,
            "model": model,
            "version": version,
            "modelHash": model_hash,
        }
        if financial_scope:
            body["financialScope"] = financial_scope
        if metadata:
            body["metadata"] = metadata
        data = await self._http.post("/agents", json=body)
        return Agent.model_validate(data["data"])

    async def get(self, agent_id: str) -> Agent:
        data = await self._http.get(f"/agents/{agent_id}")
        return Agent.model_validate(data["data"])

    async def list(
        self,
        *,
        status: str | None = None,
        limit: int = 50,
        before: str | None = None,
    ) -> tuple[list[Agent], Pagination]:
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        if before:
            params["before"] = before
        data = await self._http.get("/agents", params=params)
        agents = [Agent.model_validate(r) for r in data["data"]]
        pagination = Pagination.model_validate(data["pagination"])
        return agents, pagination

    async def update(
        self,
        agent_id: str,
        *,
        name: str | None = None,
        status: str | None = None,
        model_hash: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Agent:
        body: dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if status is not None:
            body["status"] = status
        if model_hash is not None:
            body["modelHash"] = model_hash
        if metadata is not None:
            body["metadata"] = metadata
        data = await self._http.patch(f"/agents/{agent_id}", json=body)
        return Agent.model_validate(data["data"])

    async def certify(self, agent_id: str) -> Certificate:
        data = await self._http.post(f"/agents/{agent_id}/certify")
        return Certificate.model_validate(data["data"])

    async def sign(self, agent_id: str, *, payload: dict[str, Any]) -> SignResult:
        data = await self._http.post(f"/agents/{agent_id}/sign", json={"payload": payload})
        return SignResult.model_validate(data["data"])

    async def halt(self, agent_id: str, *, reason: str | None = None) -> HaltReceipt:
        body: dict[str, Any] = {}
        if reason:
            body["reason"] = reason
        data = await self._http.post(f"/agents/{agent_id}/halt", json=body)
        return HaltReceipt.model_validate(data["data"])

    async def risk(self, agent_id: str, *, window_days: int = 7) -> AgentRisk:
        data = await self._http.get(f"/agents/{agent_id}/risk", params={"window_days": window_days})
        return AgentRisk.model_validate(data["data"])


class _EventsResource:
    def __init__(self, http: HttpTransport) -> None:
        self._http = http

    async def create(
        self,
        *,
        agent_id: str,
        action_type: str,
        chain_id: str | None = None,
        session_id: str | None = None,
        occurred_at: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> BehaviorEvent:
        body: dict[str, Any] = {
            "agentId": agent_id,
            "actionType": action_type,
        }
        if chain_id:
            body["chainId"] = chain_id
        if session_id:
            body["sessionId"] = session_id
        if occurred_at:
            body["occurredAt"] = occurred_at
        if details:
            body["details"] = details
        data = await self._http.post("/events", json=body)
        return BehaviorEvent.model_validate(data["data"])

    async def list(
        self,
        *,
        agent_id: str | None = None,
        band: RiskBand | str | None = None,
        action_type: str | None = None,
        limit: int = 50,
        before: str | None = None,
    ) -> tuple[list[BehaviorEvent], Pagination]:
        params: dict[str, Any] = {"limit": limit}
        if agent_id:
            params["agent_id"] = agent_id
        if band:
            params["band"] = band if isinstance(band, str) else band.value
        if action_type:
            params["action_type"] = action_type
        if before:
            params["before"] = before
        data = await self._http.get("/events", params=params)
        events = [BehaviorEvent.model_validate(r) for r in data["data"]]
        pagination = Pagination.model_validate(data["pagination"])
        return events, pagination


class _ChainsResource:
    def __init__(self, http: HttpTransport) -> None:
        self._http = http

    async def create(
        self,
        *,
        name: str,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DecisionChain:
        body: dict[str, Any] = {"name": name}
        if description:
            body["description"] = description
        if metadata:
            body["metadata"] = metadata
        data = await self._http.post("/chains", json=body)
        return DecisionChain.model_validate(data["data"])

    async def get(self, chain_id: str) -> DecisionChain:
        data = await self._http.get(f"/chains/{chain_id}")
        return DecisionChain.model_validate(data["data"])

    async def list(
        self,
        *,
        status: str | None = None,
        limit: int = 50,
        before: str | None = None,
    ) -> tuple[list[DecisionChain], Pagination]:
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        if before:
            params["before"] = before
        data = await self._http.get("/chains", params=params)
        chains = [DecisionChain.model_validate(r) for r in data["data"]]
        pagination = Pagination.model_validate(data["pagination"])
        return chains, pagination

    async def close(self, chain_id: str) -> DecisionChain:
        data = await self._http.post(f"/chains/{chain_id}/close")
        return DecisionChain.model_validate(data["data"])


class _ReportsResource:
    def __init__(self, http: HttpTransport) -> None:
        self._http = http

    async def create(
        self,
        *,
        agent_id: str,
        window_days: int = 30,
        notes: str | None = None,
        standards_frameworks: list[StandardsFramework | str] | None = None,
    ) -> ComplianceReport:
        body: dict[str, Any] = {
            "agentId": agent_id,
            "windowDays": window_days,
        }
        if notes:
            body["notes"] = notes
        if standards_frameworks:
            body["standardsFrameworks"] = [
                f if isinstance(f, str) else f.value for f in standards_frameworks
            ]
        data = await self._http.post("/reports/compliance", json=body)
        return ComplianceReport.model_validate(data["data"])

    async def get(self, report_id: str) -> ComplianceReport:
        data = await self._http.get(f"/reports/{report_id}")
        return ComplianceReport.model_validate(data["data"])


class _VerifyResource:
    def __init__(self, http: HttpTransport) -> None:
        # Verify endpoints are public — no auth header needed.
        # We reuse the same transport for simplicity; the server ignores the Bearer
        # token on /v1/verify/* paths.
        self._http = http

    async def message(
        self,
        *,
        agent_id: str,
        payload: dict[str, Any],
        signature: str,
        certificate_serial: str,
    ) -> MessageVerifyResult:
        body = {
            "agentId": agent_id,
            "payload": payload,
            "signature": signature,
            "certificateSerial": certificate_serial,
        }
        data = await self._http.post("/verify/message", json=body)
        return MessageVerifyResult.model_validate(data["data"])

    async def chain(self, chain_id: str) -> ChainVerifyResult:
        data = await self._http.get(f"/verify/chain/{chain_id}")
        return ChainVerifyResult.model_validate(data["data"])

    async def certificate(self, serial: str) -> dict[str, Any]:
        data = await self._http.get(f"/verify/{serial}")
        return data["data"]  # type: ignore[no-any-return]
