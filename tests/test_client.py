"""Unit tests for Kakunin SDK — all requests mocked via respx."""

from __future__ import annotations

import pytest
import respx
import httpx

from kakunin import Kakunin
from kakunin.exceptions import AuthenticationError, NotFoundError, RateLimitError, ValidationError
from kakunin.models import AgentStatus, RiskBand, ChainStatus


BASE = "https://api.kakunin.ai/v1"

AGENT_FIXTURE = {
    "id": "agt-123",
    "name": "TradeBot",
    "model": "gpt-4o",
    "version": "2024-11",
    "status": "active",
    "model_hash": "sha256:abc",
    "metadata": None,
    "created_at": "2026-05-18T00:00:00Z",
}

EVENT_FIXTURE = {
    "id": "evt-456",
    "agent_id": "agt-123",
    "action_type": "transaction_initiated",
    "risk_score": 0.1,
    "risk_band": "low",
    "occurred_at": "2026-05-18T00:00:00Z",
    "source_ip": None,
    "payload": None,
}

CHAIN_FIXTURE = {
    "chain_id": "chn-789",
    "name": "Trade #42",
    "description": None,
    "status": "open",
    "chain_hash": None,
    "event_count": 0,
    "created_at": "2026-05-18T00:00:00Z",
    "closed_at": None,
}


@pytest.fixture
def client() -> Kakunin:
    return Kakunin(api_key="kak_test_key", base_url=BASE)


# ── Agents ───────────────────────────────────────────────────────────────────

@respx.mock
@pytest.mark.asyncio
async def test_agents_create(client: Kakunin) -> None:
    respx.post(f"{BASE}/agents").mock(
        return_value=httpx.Response(200, json={"data": AGENT_FIXTURE})
    )
    agent = await client.agents.create(
        name="TradeBot",
        model="gpt-4o",
        version="2024-11",
        model_hash="sha256:abc",
    )
    assert agent.id == "agt-123"
    assert agent.status == AgentStatus.active
    assert agent.model_hash == "sha256:abc"


@respx.mock
@pytest.mark.asyncio
async def test_agents_get(client: Kakunin) -> None:
    respx.get(f"{BASE}/agents/agt-123").mock(
        return_value=httpx.Response(200, json={"data": AGENT_FIXTURE})
    )
    agent = await client.agents.get("agt-123")
    assert agent.name == "TradeBot"


@respx.mock
@pytest.mark.asyncio
async def test_agents_list(client: Kakunin) -> None:
    respx.get(f"{BASE}/agents").mock(
        return_value=httpx.Response(200, json={
            "data": [AGENT_FIXTURE],
            "pagination": {"limit": 50, "has_next_page": False, "next_cursor": None},
        })
    )
    agents, pagination = await client.agents.list()
    assert len(agents) == 1
    assert not pagination.has_next_page


@respx.mock
@pytest.mark.asyncio
async def test_agents_halt(client: Kakunin) -> None:
    halt_fixture = {
        "agent_id": "agt-123",
        "halt_event_id": "hlt-001",
        "certificate_serial": "DEADBEEF",
        "halted_at": "2026-05-18T00:00:00Z",
        "signature": "base64sig==",
        "algorithm": "RSASSA_PKCS1_V1_5_SHA_256",
    }
    respx.post(f"{BASE}/agents/agt-123/halt").mock(
        return_value=httpx.Response(200, json={"data": halt_fixture})
    )
    receipt = await client.agents.halt("agt-123", reason="Audit triggered")
    assert receipt.agent_id == "agt-123"
    assert receipt.signature == "base64sig=="


# ── Events ───────────────────────────────────────────────────────────────────

@respx.mock
@pytest.mark.asyncio
async def test_events_create(client: Kakunin) -> None:
    respx.post(f"{BASE}/events").mock(
        return_value=httpx.Response(200, json={"data": {
            "event_id": "evt-456",
            "risk_score": 0.1,
            "risk_band": "low",
            "action_type": "transaction_initiated",
            "occurred_at": "2026-05-18T00:00:00Z",
            "revocation_check_queued": False,
        }})
    )
    # events.create returns raw BehaviorEvent shape — map what the API actually returns
    resp = await client._http.post("/events", json={
        "agentId": "agt-123",
        "actionType": "transaction_initiated",
    })
    assert resp["data"]["risk_band"] == "low"


@respx.mock
@pytest.mark.asyncio
async def test_events_list(client: Kakunin) -> None:
    respx.get(f"{BASE}/events").mock(
        return_value=httpx.Response(200, json={
            "data": [EVENT_FIXTURE],
            "pagination": {"limit": 50, "has_next_page": False, "next_cursor": None},
        })
    )
    events, pagination = await client.events.list(band=RiskBand.low)
    assert events[0].risk_band == RiskBand.low


# ── Chains ───────────────────────────────────────────────────────────────────

@respx.mock
@pytest.mark.asyncio
async def test_chains_create(client: Kakunin) -> None:
    respx.post(f"{BASE}/chains").mock(
        return_value=httpx.Response(200, json={"data": CHAIN_FIXTURE})
    )
    chain = await client.chains.create(name="Trade #42")
    assert chain.chain_id == "chn-789"
    assert chain.status == ChainStatus.open


@respx.mock
@pytest.mark.asyncio
async def test_chains_close(client: Kakunin) -> None:
    closed = {**CHAIN_FIXTURE, "status": "closed", "chain_hash": "sha256hash", "event_count": 3}
    respx.post(f"{BASE}/chains/chn-789/close").mock(
        return_value=httpx.Response(200, json={"data": closed})
    )
    chain = await client.chains.close("chn-789")
    assert chain.status == ChainStatus.closed
    assert chain.chain_hash == "sha256hash"


# ── Verify ───────────────────────────────────────────────────────────────────

@respx.mock
@pytest.mark.asyncio
async def test_verify_chain_verified(client: Kakunin) -> None:
    respx.get(f"{BASE}/verify/chain/chn-789").mock(
        return_value=httpx.Response(200, json={"data": {
            "chain_id": "chn-789",
            "name": "Trade #42",
            "status": "closed",
            "chain_hash": "sha256hash",
            "event_count": 3,
            "verified": True,
            "created_at": "2026-05-18T00:00:00Z",
            "closed_at": "2026-05-18T01:00:00Z",
        }})
    )
    result = await client.verify.chain("chn-789")
    assert result.verified is True


# ── Error mapping ─────────────────────────────────────────────────────────────

@respx.mock
@pytest.mark.asyncio
async def test_raises_auth_error(client: Kakunin) -> None:
    respx.get(f"{BASE}/agents/agt-x").mock(
        return_value=httpx.Response(401, json={"error": "Invalid or revoked API key"})
    )
    with pytest.raises(AuthenticationError):
        await client.agents.get("agt-x")


@respx.mock
@pytest.mark.asyncio
async def test_raises_not_found(client: Kakunin) -> None:
    respx.get(f"{BASE}/agents/missing").mock(
        return_value=httpx.Response(404, json={"error": "Agent not found"})
    )
    with pytest.raises(NotFoundError):
        await client.agents.get("missing")


@respx.mock
@pytest.mark.asyncio
async def test_raises_rate_limit(client: Kakunin) -> None:
    respx.get(f"{BASE}/agents").mock(
        return_value=httpx.Response(
            429,
            json={"error": "Rate limit exceeded"},
            headers={"Retry-After": "5"},
        )
    )
    with pytest.raises(RateLimitError) as exc_info:
        await client.agents.list()
    assert exc_info.value.retry_after == 5


@respx.mock
@pytest.mark.asyncio
async def test_raises_validation_error(client: Kakunin) -> None:
    respx.post(f"{BASE}/agents").mock(
        return_value=httpx.Response(400, json={"error": "modelHash is required"})
    )
    with pytest.raises(ValidationError):
        await client.agents.create(
            name="X", model="gpt-4o", version="1", model_hash=""
        )
