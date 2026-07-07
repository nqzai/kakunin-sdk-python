"""Pydantic response models mirroring the Kakunin API schema."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AgentStatus(str, Enum):
    active = "active"
    suspended = "suspended"
    retired = "retired"


class CertificateStatus(str, Enum):
    active = "active"
    revoked = "revoked"
    expired = "expired"


class RiskBand(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class DriftBand(str, Enum):
    stable = "stable"
    moderate = "moderate"
    significant = "significant"


class ChainStatus(str, Enum):
    open = "open"
    closed = "closed"


class ReportStatus(str, Enum):
    generating = "generating"
    ready = "ready"
    failed = "failed"


class StandardsFramework(str, Enum):
    iso_27001 = "iso_27001"
    nist_csf = "nist_csf"


class Pagination(BaseModel):
    limit: int
    has_next_page: bool
    next_cursor: str | None = None


class FinancialScope(BaseModel):
    max_single_trade_usd: float | None = None
    daily_limit_usd: float | None = None
    permitted_instruments: list[str] = Field(default_factory=list)
    permitted_venues: list[str] = Field(default_factory=list)
    leverage_permitted: bool = False
    max_leverage_ratio: float | None = None


class Agent(BaseModel):
    id: str
    name: str
    model: str
    version: str
    status: AgentStatus
    model_hash: str | None = None
    metadata: dict[str, Any] | None = None
    created_at: datetime


class Certificate(BaseModel):
    id: str
    agent_id: str
    serial_number: str
    status: CertificateStatus
    pem: str | None = None
    issued_at: datetime
    expires_at: datetime
    halt_receipt: dict[str, Any] | None = None


class BehaviorEvent(BaseModel):
    id: str
    agent_id: str
    action_type: str
    risk_score: float
    risk_band: RiskBand
    occurred_at: datetime
    source_ip: str | None = None
    payload: dict[str, Any] | None = None


class ChainEvent(BaseModel):
    sequence: int
    event_id: str
    agent_id: str
    agent_name: str | None = None
    model_hash: str | None = None
    certificate_serial: str | None = None
    action_type: str
    risk_score: float
    risk_band: RiskBand
    occurred_at: datetime
    details: dict[str, Any] | None = None


class ParticipatingAgent(BaseModel):
    agent_id: str
    agent_name: str | None = None
    model_hash: str | None = None
    certificate_serial: str | None = None
    certificate_status: CertificateStatus | None = None


class DecisionChain(BaseModel):
    chain_id: str
    name: str
    description: str | None = None
    status: ChainStatus
    chain_hash: str | None = None
    event_count: int
    created_at: datetime
    closed_at: datetime | None = None
    events: list[ChainEvent] | None = None
    participating_agents: list[ParticipatingAgent] | None = None
    metadata: dict[str, Any] | None = None


class ComplianceReport(BaseModel):
    id: str
    agent_id: str
    title: str
    status: ReportStatus
    summary: str | None = None
    period_start: datetime
    period_end: datetime
    created_at: datetime
    updated_at: datetime


class HaltReceipt(BaseModel):
    agent_id: str
    halt_event_id: str
    certificate_serial: str | None = None
    halted_at: str
    signature: str
    algorithm: str


class SignResult(BaseModel):
    signature: str
    algorithm: str
    certificate_serial: str | None = None
    payload_hash: str
    signed_at: datetime


class MessageVerifyResult(BaseModel):
    valid: bool
    certificate_status: CertificateStatus | None = None
    agent_id: str | None = None
    agent_name: str | None = None
    model_hash: str | None = None


class ChainVerifyResult(BaseModel):
    chain_id: str
    name: str
    status: ChainStatus
    chain_hash: str | None = None
    event_count: int | None = None
    verified: bool | None = None
    note: str | None = None
    created_at: datetime | None = None
    closed_at: datetime | None = None


class DriftInfo(BaseModel):
    drift_score: float | None = None
    drift_band: DriftBand | None = None
    contributing_factors: list[str] | None = None
    computed_at: datetime | None = None
    drift_trend: str | None = None
    baseline_established_at: datetime | None = None
    baseline_events_analyzed: int | None = None
    note: str | None = None


class AgentRisk(BaseModel):
    agent_id: str
    risk_score: float
    risk_band: RiskBand
    event_count: int
    high_risk_count: int
    medium_risk_count: int
    avg_risk_score: float
    drift: DriftInfo | None = None
    window_days: int
    computed_at: datetime
