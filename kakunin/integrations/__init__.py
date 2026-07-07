"""Framework integration shims for Kakunin.

Two layers of integration:

1. **Scope verification** (pre-execution guard — raises ScopeViolationError):
   - kakunin.integrations.scope     — verify_agent_scope (any framework)
   - kakunin.integrations.langchain — KakuninToolGuard, langchain_scope_callback
   - kakunin.integrations.llamaindex — KakuninFunctionToolGuard
   - kakunin.integrations.google_antigravity — KakuninPreToolCallDecideHook

2. **Behavioral event emission** (fire-and-forget monitoring — never raises):
   - kakunin.integrations.langgraph — kakunin_node decorator
   - kakunin.integrations.crewai    — KakuninCrewAgent (also supports required_scopes)
   - kakunin.integrations.autogen   — KakuninConversableAgent (also supports required_scopes)
   - kakunin.integrations.google_antigravity — session, turn, tool, and error lifecycle hooks

3. **CAMEL-AI** (toolkit + monitored agent — requires camel-ai>=0.2.0):
   - kakunin.integrations.camel — KakuninToolkit (4 FunctionTools), KakuninCamelAgent
"""

from .google_antigravity import (
    KakuninSessionStartHook,
    KakuninSessionEndHook,
    KakuninPreTurnHook,
    KakuninPostTurnHook,
    KakuninPreToolCallDecideHook,
    KakuninPostToolCallHook,
    KakuninOnToolErrorHook,
    get_kakunin_hooks,
)
from .openai_swarm import KakuninSwarm
from .openai_assistants import handle_assistants_requires_action

__all__ = [
    "KakuninSessionStartHook",
    "KakuninSessionEndHook",
    "KakuninPreTurnHook",
    "KakuninPostTurnHook",
    "KakuninPreToolCallDecideHook",
    "KakuninPostToolCallHook",
    "KakuninOnToolErrorHook",
    "get_kakunin_hooks",
    "KakuninSwarm",
    "handle_assistants_requires_action",
]


