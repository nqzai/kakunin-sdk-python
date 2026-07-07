"""Kakunin Python SDK — AI agent compliance infrastructure."""

from .client import Kakunin
from .exceptions import KakuninError, AuthenticationError, NotFoundError, RateLimitError, ValidationError, ScopeViolationError
from .integrations.scope import verify_agent_scope
from .models import (
    Agent,
    AgentStatus,
    Certificate,
    CertificateStatus,
    BehaviorEvent,
    RiskBand,
    DecisionChain,
    ChainStatus,
    ComplianceReport,
    ReportStatus,
    DriftBand,
    HaltReceipt,
    SignResult,
    MessageVerifyResult,
    Pagination,
    StandardsFramework,
)

__all__ = [
    "Kakunin",
    "KakuninError",
    "AuthenticationError",
    "NotFoundError",
    "RateLimitError",
    "ValidationError",
    "ScopeViolationError",
    "verify_agent_scope",
    "Agent",
    "AgentStatus",
    "Certificate",
    "CertificateStatus",
    "BehaviorEvent",
    "RiskBand",
    "DecisionChain",
    "ChainStatus",
    "ComplianceReport",
    "ReportStatus",
    "DriftBand",
    "HaltReceipt",
    "SignResult",
    "MessageVerifyResult",
    "Pagination",
    "StandardsFramework",
]

__version__ = "0.1.0"
