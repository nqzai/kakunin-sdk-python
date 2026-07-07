"""Kakunin SDK exception hierarchy."""

from __future__ import annotations


class KakuninError(Exception):
    """Base exception for all Kakunin SDK errors."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class AuthenticationError(KakuninError):
    """Invalid or revoked API key (HTTP 401)."""


class ValidationError(KakuninError):
    """Request body failed validation (HTTP 400)."""


class NotFoundError(KakuninError):
    """Resource not found (HTTP 404)."""


class ConflictError(KakuninError):
    """Resource state conflict — e.g. chain already closed (HTTP 409)."""


class UnprocessableError(KakuninError):
    """Business rule violation — e.g. retired agent, missing model_hash (HTTP 422)."""


class RateLimitError(KakuninError):
    """Rate limit exceeded (HTTP 429). Check retry_after for backoff seconds."""

    def __init__(self, message: str, retry_after: int | None = None) -> None:
        super().__init__(message, status_code=429)
        self.retry_after = retry_after


class ServerError(KakuninError):
    """Unexpected server error (HTTP 5xx)."""


class ScopeViolationError(KakuninError):
    """
    Agent is not authorised to execute the guarded operation.

    Raised by verify_agent_scope when:
    - Agent status is not "active" (suspended / retired)
    - Agent metadata lacks one or more required_scopes
    """

    def __init__(
        self,
        message: str,
        agent_id: str | None = None,
        agent_status: str | None = None,
        missing_scopes: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.agent_id = agent_id
        self.agent_status = agent_status
        self.missing_scopes = missing_scopes or []
