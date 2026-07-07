"""
OpenAI Swarm integration shim.

Provides KakuninSwarm — a subclass of Swarm that automatically enforces scope verification
and logs behavioral events to Kakunin.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

try:
    from swarm import Swarm, Agent  # type: ignore[import-not-found]
    HAS_SWARM = True
except ImportError:
    # Dummy fallbacks
    class Swarm: pass  # type: ignore[no-redef]
    class Agent: pass  # type: ignore[no-redef]
    HAS_SWARM = False

from .scope import _check_scope
from ..client import Kakunin


class KakuninSwarm(Swarm):  # type: ignore[misc]
    """
    Subclass of OpenAI Swarm that automatically enforces scope verification
    and logs behavioral events to Kakunin.
    """

    def __init__(
        self,
        kakunin: Kakunin,
        agent_id: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._kakunin = kakunin
        self._agent_id = agent_id

    def _emit(self, action_type: str, details: dict[str, Any] | None = None) -> None:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self._kakunin.events.create(
                    agent_id=self._agent_id,
                    action_type=action_type,
                    details=details or {},
                ))
            else:
                loop.run_until_complete(self._kakunin.events.create(
                    agent_id=self._agent_id,
                    action_type=action_type,
                    details=details or {},
                ))
        except Exception:
            pass

    def run(
        self,
        agent: Any,
        messages: list[Any],
        tool_scopes_mapping: dict[str, list[str]] | None = None,
        chain_id: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Runs the Swarm agent, dynamically wrapping its functions with Kakunin scope validation.
        """
        self._emit("session_started", {"chain_id": chain_id} if chain_id else None)

        # Save original functions to restore later
        original_functions = list(agent.functions) if hasattr(agent, "functions") else []
        tool_scopes = tool_scopes_mapping or {}

        # Wrap agent functions with Kakunin scope checks and tracking
        wrapped_functions = []
        for fn in original_functions:
            fn_name = fn.__name__
            required_scopes = tool_scopes.get(fn_name)

            def create_wrapped_fn(f: Callable[..., Any], req_scopes: list[str] | None) -> Callable[..., Any]:
                if asyncio.iscoroutinefunction(f):
                    async def async_wrapped(*args_inner: Any, **kwargs_inner: Any) -> Any:
                        try:
                            # Status/Active check is always ran, even if req_scopes is None/empty
                            await _check_scope(self._kakunin, self._agent_id, req_scopes)
                        except Exception as e:
                            self._emit("unauthorized_access_attempt", {"tool_name": f.__name__, "error": str(e)})
                            raise e

                        try:
                            res = await f(*args_inner, **kwargs_inner)
                            self._emit("tool_executed", {"tool_name": f.__name__})
                            return res
                        except Exception as e:
                            self._emit("tool_error", {"tool_name": f.__name__, "error": str(e)})
                            raise e
                    # Copy metadata
                    async_wrapped.__name__ = f.__name__
                    async_wrapped.__doc__ = f.__doc__
                    return async_wrapped
                else:
                    def sync_wrapped(*args_inner: Any, **kwargs_inner: Any) -> Any:
                        try:
                            try:
                                loop = asyncio.get_running_loop()
                                import concurrent.futures
                                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                                    fut = pool.submit(asyncio.run, _check_scope(self._kakunin, self._agent_id, req_scopes))
                                    fut.result()
                            except RuntimeError:
                                asyncio.run(_check_scope(self._kakunin, self._agent_id, req_scopes))
                        except Exception as e:
                            self._emit("unauthorized_access_attempt", {"tool_name": f.__name__, "error": str(e)})
                            raise e

                        try:
                            res = f(*args_inner, **kwargs_inner)
                            self._emit("tool_executed", {"tool_name": f.__name__})
                            return res
                        except Exception as e:
                            self._emit("tool_error", {"tool_name": f.__name__, "error": str(e)})
                            raise e
                    # Copy metadata
                    sync_wrapped.__name__ = f.__name__
                    sync_wrapped.__doc__ = f.__doc__
                    return sync_wrapped

            wrapped_functions.append(create_wrapped_fn(fn, required_scopes))

        agent.functions = wrapped_functions

        try:
            # Emit prompt log
            if messages:
                last_msg = messages[-1]
                prompt = last_msg.get("content") if isinstance(last_msg, dict) else getattr(last_msg, "content", None)
                if prompt:
                    self._emit("prompt_received", {"prompt": prompt})

            response = super().run(agent=agent, messages=messages, **kwargs)

            # Emit response log
            if hasattr(response, "messages") and response.messages:
                last_resp = response.messages[-1]
                resp_content = last_resp.get("content") if isinstance(last_resp, dict) else getattr(last_resp, "content", None)
                if resp_content:
                    self._emit("response_generated", {"response": resp_content})

            return response
        finally:
            # Restore original functions on the agent object
            agent.functions = original_functions
