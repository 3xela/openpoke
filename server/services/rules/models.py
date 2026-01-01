# server/services/rules/models.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional


class RuleScope(str, Enum):
    """
    Where a rule applies.

    - global: always injected/enforced
    - chat: interactive chat requests
    - email: email watcher / email-related flows
    """
    GLOBAL = "global"
    CHAT = "chat"
    EMAIL = "email"


class RuleActionType(str, Enum):
    """
    Minimal v1 action types.

    v1 focuses on:
      - injecting preferences into prompts (soft)
      - blocking / requiring confirmation for tools (hard)
      - optional routing nudges (soft)
    """
    PROMPT_INJECT = "prompt_inject"   # payload: {"text": str}
    BLOCK_TOOL = "block_tool"         # payload: {"tools": [str], "reason": Optional[str]}
    CONFIRM_TOOL = "confirm_tool"     # payload: {"tools": [str], "reason": Optional[str]}

    # Optional (can implement in v2; keep in schema now if you want)
    BOOST_AGENT = "boost_agent"       # payload: {"agents": [str], "boost": float}
    EXCLUDE_AGENT = "exclude_agent"   # payload: {"agents": [str]}
    FORCE_AGENT = "force_agent"       # payload: {"agents": [str]}


@dataclass(frozen=True)
class RuleAction:
    type: RuleActionType
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Rule:
    """
    A single persisted rule.

    Keep it small, explicit, and deterministic.
    """
    id: str
    enabled: bool = True
    scope: RuleScope = RuleScope.GLOBAL

    # The original user text that created the rule (for display / debugging)
    raw_text: str = ""

    # One rule can produce multiple actions
    actions: List[RuleAction] = field(default_factory=list)

    # Optional metadata (store.py will set if present)
    created_at: Optional[str] = None  # ISO-8601 UTC
    updated_at: Optional[str] = None  # ISO-8601 UTC


@dataclass(frozen=True)
class RuleParseResult:
    """
    Output of parser_agent.parse_user_rule(text).

    - If needs_confirmation is True, you should ask the user to confirm before saving.
    """
    rule: Rule
    explanation: str
    confidence: float = 1.0
    needs_confirmation: bool = True


@dataclass(frozen=True)
class ToolDecision:
    """
    Result of rule enforcement before a tool call.
    """
    allowed: bool
    requires_confirmation: bool = False
    block_reason: Optional[str] = None
    confirm_reason: Optional[str] = None

    @staticmethod
    def allow() -> "ToolDecision":
        return ToolDecision(allowed=True)

    @staticmethod
    def block(reason: str) -> "ToolDecision":
        return ToolDecision(allowed=False, block_reason=reason)

    @staticmethod
    def require_confirm(reason: str) -> "ToolDecision":
        return ToolDecision(allowed=True, requires_confirmation=True, confirm_reason=reason)
