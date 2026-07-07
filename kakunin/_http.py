"""Internal HTTP transport — wraps httpx, handles auth + error mapping."""

from __future__ import annotations

from typing import Any

import httpx

from .exceptions import (
    AuthenticationError,
    ConflictError,
    KakuninError,
    NotFoundError,
    RateLimitError,
    ServerError,
    UnprocessableError,
    ValidationError,
)

DEFAULT_BASE_URL = "https://api.kakunin.ai/v1"
DEFAULT_TIMEOUT = 30.0


def _raise_for_status(response: httpx.Response) -> None:
    if response.is_success:
        return

    try:
        body = response.json()
        message = body.get("error", response.text)
    except Exception:
        message = response.text

    status = response.status_code

    if status == 400:
        raise ValidationError(message, status_code=status)
    if status == 401:
        raise AuthenticationError(message, status_code=status)
    if status == 404:
        raise NotFoundError(message, status_code=status)
    if status == 409:
        raise ConflictError(message, status_code=status)
    if status == 422:
        raise UnprocessableError(message, status_code=status)
    if status == 429:
        retry_after_raw = response.headers.get("Retry-After")
        retry_after = int(retry_after_raw) if retry_after_raw else None
        raise RateLimitError(message, retry_after=retry_after)
    if status >= 500:
        raise ServerError(message, status_code=status)

    raise KakuninError(message, status_code=status)


class HttpTransport:
    def __init__(self, api_key: str, base_url: str, timeout: float) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "kakunin-python/0.1.0",
            },
            timeout=timeout,
        )

    async def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        response = await self._client.get(path, params=params)
        _raise_for_status(response)
        return response.json()

    async def post(self, path: str, json: dict[str, Any] | None = None) -> Any:
        response = await self._client.post(path, json=json or {})
        _raise_for_status(response)
        return response.json()

    async def patch(self, path: str, json: dict[str, Any] | None = None) -> Any:
        response = await self._client.patch(path, json=json or {})
        _raise_for_status(response)
        return response.json()

    async def delete(self, path: str) -> Any:
        response = await self._client.delete(path)
        _raise_for_status(response)
        return response.json()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "HttpTransport":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()
