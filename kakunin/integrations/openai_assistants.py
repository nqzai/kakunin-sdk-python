"""
OpenAI Assistants API integration helper.

Provides handle_assistants_requires_action to easily validate scopes, execute functions,
and report behavioral events/anomalies to Kakunin.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable

from .scope import _check_scope
from ..client import Kakunin


async def handle_assistants_requires_action(
    kakunin: Kakunin,
    agent_id: str,
    run: Any,
    tool_scopes_mapping: dict[str, list[str]],
    tool_funcs: dict[str, Callable[..., Any]],
    catch_exceptions: bool = True,
) -> list[dict[str, Any]]:
    """
    Validates agent status and scopes, executes target tool functions, logs events to Kakunin,
    and formats tool outputs to be submitted back to OpenAI.

    Usage::

        from kakunin.integrations.openai_assistants import handle_assistants_requires_action

        if run.status == "requires_action":
            tool_outputs = await handle_assistants_requires_action(
                kakunin=client,
                agent_id="agt-123",
                run=run,
                tool_scopes_mapping={"execute_trade": ["trade.execute"]},
                tool_funcs={"execute_trade": execute_trade_function},
            )
            openai_client.beta.threads.runs.submit_tool_outputs(
                thread_id=thread.id,
                run_id=run.id,
                tool_outputs=tool_outputs,
            )
    """
    tool_outputs = []
    
    # Verify that the run requires action and has submit_tool_outputs
    required_action = getattr(run, "required_action", None)
    if not required_action:
        return []

    submit_tool_outputs = getattr(required_action, "submit_tool_outputs", None)
    if not submit_tool_outputs:
        return []

    tool_calls = getattr(submit_tool_outputs, "tool_calls", [])
    if not tool_calls:
        return []

    def _emit(action_type: str, details: dict[str, Any] | None = None) -> None:
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(kakunin.events.create(
                agent_id=agent_id,
                action_type=action_type,
                details=details or {},
            ))
        except Exception:
            pass

    for tool_call in tool_calls:
        tool_call_id = getattr(tool_call, "id", None)
        function_ctx = getattr(tool_call, "function", None)
        if not tool_call_id or not function_ctx:
            continue

        tool_name = getattr(function_ctx, "name", "")
        arguments_str = getattr(function_ctx, "arguments", "{}")
        
        try:
            arguments = json.loads(arguments_str)
        except Exception:
            arguments = {}

        required_scopes = tool_scopes_mapping.get(tool_name)
        
        # 1. Enforce active status and scope verification
        try:
            await _check_scope(kakunin, agent_id, required_scopes)
        except Exception as e:
            _emit("unauthorized_access_attempt", {"tool_name": tool_name, "error": str(e)})
            if catch_exceptions:
                tool_outputs.append({
                    "tool_call_id": tool_call_id,
                    "output": json.dumps({"error": f"Scope violation check failed: {e}"})
                })
                continue
            raise e

        # 2. Execute target function
        func = tool_funcs.get(tool_name)
        if not func:
            error_msg = f"Tool function '{tool_name}' not registered in tool_funcs mapping"
            _emit("tool_error", {"tool_name": tool_name, "error": error_msg})
            tool_outputs.append({
                "tool_call_id": tool_call_id,
                "output": json.dumps({"error": error_msg})
            })
            continue

        try:
            if asyncio.iscoroutinefunction(func):
                res = await func(**arguments)
            else:
                res = func(**arguments)
            
            # Formats output as string
            output_data = res if isinstance(res, str) else json.dumps(res)
            _emit("tool_executed", {"tool_name": tool_name})
            tool_outputs.append({
                "tool_call_id": tool_call_id,
                "output": output_data
            })
        except Exception as e:
            error_msg = str(e)
            _emit("tool_error", {"tool_name": tool_name, "error": error_msg})
            if catch_exceptions:
                tool_outputs.append({
                    "tool_call_id": tool_call_id,
                    "output": json.dumps({"error": f"Tool execution failed: {error_msg}"})
                })
            else:
                raise e

    return tool_outputs
