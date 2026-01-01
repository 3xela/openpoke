from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set

from .models import Rule, RuleAction, RuleActionType, RuleScope, ToolDecision


@dataclass(frozen=True)
class AppliedRuleContext:
    """
    Convenience bundle returned by collect_effective_rules().
    """
    rules: List[Rule]
    prompt_instructions: str
    blocked_tools: Set[str]
    confirm_tools: Set[str]
    blocked_reasons: Dict[str, str]
    confirm_reasons: Dict[str, str]


def _iter_actions(rule: Rule) -> Iterable[RuleAction]:
    actions = getattr(rule, "actions", None)
    if not actions:
        return []
    return actions


def _rule_enabled(rule: Rule) -> bool:
    return bool(getattr(rule, "enabled", True))


def _scope_matches(rule_scope: RuleScope, desired: RuleScope) -> bool:
    # global rules apply everywhere
    return rule_scope == RuleScope.GLOBAL or rule_scope == desired


def filter_rules_for_scope(rules: List[Rule], scope: RuleScope) -> List[Rule]:
    out: List[Rule] = []
    for r in rules:
        if not _rule_enabled(r):
            continue
        rs = getattr(r, "scope", RuleScope.GLOBAL)
        if _scope_matches(rs, scope):
            out.append(r)
    return out


def build_prompt_instructions(rules: List[Rule], scope: RuleScope) -> str:
    """
    Returns a deterministic prompt block to inject into InteractionAgent/system prompts.

    v1 behavior:
      - Includes RuleActionType.PROMPT_INJECT payload texts
      - Also includes the raw_text of rules that have PROMPT_INJECT actions, as fallback
    """
    effective = filter_rules_for_scope(rules, scope)

    lines: List[str] = []
    for r in effective:
        injected_any = False
        for a in _iter_actions(r):
            if a.type != RuleActionType.PROMPT_INJECT:
                continue
            text = (a.payload or {}).get("text", "")
            if isinstance(text, str) and text.strip():
                lines.append(text.strip())
                injected_any = True

        # Fallback: if rule intended prompt injection but forgot payload text
        if not injected_any:
            # If this rule has no actions or is a "prompt rule" stored only as raw_text,
            # you can choose to include it here. For v1, keep conservative:
            # only include raw_text if it "looks like" a preference (short).
            raw = (getattr(r, "raw_text", "") or "").strip()
            if raw and len(raw) <= 200:
                # Avoid repeating tool-block/confirm rules as prompt instructions unless you explicitly want that.
                has_tool_action = any(
                    a.type in (RuleActionType.BLOCK_TOOL, RuleActionType.CONFIRM_TOOL)
                    for a in _iter_actions(r)
                )
                if not has_tool_action:
                    lines.append(raw)

    if not lines:
        return ""

    # Deterministic, compact block.
    rendered = "\n".join(f"- {ln}" for ln in lines)
    return (
        "USER RULES (must follow):\n"
        f"{rendered}\n"
    )


def collect_effective_rules(rules: List[Rule], scope: RuleScope) -> AppliedRuleContext:
    """
    Collect the effective rules for a given scope and precompute:
      - prompt instructions
      - blocked/confirm tool sets + reasons
    """
    effective = filter_rules_for_scope(rules, scope)
    prompt = build_prompt_instructions(effective, scope)

    blocked: Set[str] = set()
    confirm: Set[str] = set()
    blocked_reasons: Dict[str, str] = {}
    confirm_reasons: Dict[str, str] = {}

    for r in effective:
        for a in _iter_actions(r):
            payload = a.payload or {}
            reason = payload.get("reason") or getattr(r, "raw_text", "") or "Blocked by user rule."
            if not isinstance(reason, str) or not reason.strip():
                reason = "Blocked by user rule."

            if a.type == RuleActionType.BLOCK_TOOL:
                tools = payload.get("tools", [])
                if isinstance(tools, str):
                    tools = [tools]
                if isinstance(tools, list):
                    for t in tools:
                        if isinstance(t, str) and t.strip():
                            blocked.add(t)
                            # first reason wins to keep stable
                            blocked_reasons.setdefault(t, reason.strip())

            elif a.type == RuleActionType.CONFIRM_TOOL:
                tools = payload.get("tools", [])
                if isinstance(tools, str):
                    tools = [tools]
                if isinstance(tools, list):
                    for t in tools:
                        if isinstance(t, str) and t.strip():
                            confirm.add(t)
                            confirm_reasons.setdefault(t, reason.strip())

    return AppliedRuleContext(
        rules=effective,
        prompt_instructions=prompt,
        blocked_tools=blocked,
        confirm_tools=confirm,
        blocked_reasons=blocked_reasons,
        confirm_reasons=confirm_reasons,
    )


def _is_confirmed(tool_args: Dict[str, Any]) -> bool:
    """
    Convention: allow any of these boolean flags to count as confirmation.

    You can standardize on one key later (e.g. confirmed=True).
    """
    for key in ("confirmed", "confirm", "user_confirmed", "approved", "allow"):
        v = tool_args.get(key, None)
        if v is True:
            return True
        if isinstance(v, str) and v.strip().lower() in ("true", "yes", "y", "1"):
            return True
        if isinstance(v, int) and v == 1:
            return True
    return False


def check_tool_call(
    rules: List[Rule],
    scope: RuleScope,
    tool_name: str,
    tool_args: Optional[Dict[str, Any]] = None,
) -> ToolDecision:
    """
    Enforce rules *right before* executing a tool.

    Precedence:
      1) block_tool
      2) confirm_tool (requires explicit confirmation flag)

    Returns:
      - ToolDecision.block(...) if blocked
      - ToolDecision.require_confirm(...) if confirmation required but not present
      - ToolDecision.allow() otherwise
    """
    tool_args = tool_args or {}
    ctx = collect_effective_rules(rules, scope)

    if tool_name in ctx.blocked_tools:
        reason = ctx.blocked_reasons.get(tool_name, "Blocked by user rule.")
        return ToolDecision.block(reason)

    if tool_name in ctx.confirm_tools and not _is_confirmed(tool_args):
        reason = ctx.confirm_reasons.get(tool_name, "This action requires confirmation by user rule.")
        return ToolDecision.require_confirm(reason)

    return ToolDecision.allow()
