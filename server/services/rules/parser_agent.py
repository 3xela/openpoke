# server/services/rules/parser_agent.py
from __future__ import annotations

import re
import uuid
from typing import Optional

from .models import Rule, RuleAction, RuleActionType, RuleParseResult, RuleScope


# ---- Canonical tool names (from your gmail tools registry) ----
TOOL_EXECUTE_DRAFT = "gmail_execute_draft"
TOOL_FORWARD_EMAIL = "gmail_forward_email"
TOOL_DELETE_DRAFT = "gmail_delete_draft"
TOOL_CREATE_DRAFT = "gmail_create_draft"

# You can expand later.
_SEND_TOOLS = {TOOL_EXECUTE_DRAFT}
_FORWARD_TOOLS = {TOOL_FORWARD_EMAIL}
_DELETE_TOOLS = {TOOL_DELETE_DRAFT}


def _new_rule_id() -> str:
    return f"rule_{uuid.uuid4().hex[:10]}"


def _norm(s: str) -> str:
    s = s.strip().lower()

    # normalize common contractions BEFORE punctuation stripping
    s = s.replace("don't", "dont").replace("don’t", "dont")

    s = re.sub(r"[^\w\s@.-]+", " ", s)  # drop punctuation
    s = re.sub(r"\s+", " ", s).strip()
    return s


def looks_like_rule(text: str) -> bool:
    raw = text.strip().lower()
    if raw.startswith(("rule:", "rules:")):   # check BEFORE _norm
        return True

    t = _norm(text)
    if not t:
        return False

    triggers = (
        "always ",
        "never ",
        "from now on",
        "in the future",
        "please always",
        "please never",
        "dont ",        # keep only normalized form
        "do not ",
        "stop ",
        "avoid ",
        "make sure ",
    )
    return any(tr in t for tr in triggers)



def _strip_rule_prefix(text: str) -> str:
    t = text.strip()
    # allow "rule: ..." / "rules: ..."
    m = re.match(r"^\s*rules?\s*:\s*(.*)$", t, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return t


def parse_user_rule(text: str, scope: RuleScope = RuleScope.GLOBAL) -> Optional[RuleParseResult]:
    """
    Deterministic v1 template parser.

    Supported categories:
      - Block sending (execute draft)
      - Block forwarding
      - Require confirmation for sending (execute draft)
      - Block deleting drafts
      - Prompt preferences: be concise, no emoji

    Returns None if text doesn't look like a rule or doesn't match supported templates.
    """
    if not looks_like_rule(text):
        return None

    raw = _strip_rule_prefix(text)
    t = _norm(raw)

    # --- intent keywords ---
    is_never = t.startswith(("never ", "dont ", "don't ", "do not ", "stop ", "avoid "))
    is_always = t.startswith(("always ", "please always ", "make sure "))

    wants_confirm = any(k in t for k in ("confirm", "confirmation", "ask me", "approve", "approval"))
    wants_draft_first = any(k in t for k in ("show draft", "draft first", "preview", "before sending"))

    mentions_send = "send" in t and ("email" in t or "emails" in t or "mail" in t)
    mentions_forward = "forward" in t
    mentions_delete = "delete" in t and "draft" in t

    # --- prompt preference rules (soft) ---
    # Keep these very simple in v1.
    if any(k in t for k in ("be concise", "concise")) and not (mentions_send or mentions_forward or mentions_delete):
        rule = Rule(
            id=_new_rule_id(),
            enabled=True,
            scope=scope,
            raw_text=raw,
            actions=[RuleAction(type=RuleActionType.PROMPT_INJECT, payload={"text": "Be concise."})],
        )
        return RuleParseResult(
            rule=rule,
            explanation="I’ll add a rule to be concise in responses.",
            confidence=0.9,
            needs_confirmation=False,
        )

    if any(k in t for k in ("no emoji", "no emojis", "avoid emoji", "avoid emojis")) and not (
        mentions_send or mentions_forward or mentions_delete
    ):
        rule = Rule(
            id=_new_rule_id(),
            enabled=True,
            scope=scope,
            raw_text=raw,
            actions=[RuleAction(type=RuleActionType.PROMPT_INJECT, payload={"text": "Do not use emojis."})],
        )
        return RuleParseResult(
            rule=rule,
            explanation="I’ll add a rule to avoid emojis.",
            confidence=0.9,
            needs_confirmation=False,
        )

    # --- tool gating rules (hard) ---
    # 1) "never send emails" => block execute draft
    if mentions_send and is_never and not wants_confirm and not wants_draft_first:
        rule = Rule(
            id=_new_rule_id(),
            enabled=True,
            scope=scope,
            raw_text=raw,
            actions=[
                RuleAction(
                    type=RuleActionType.BLOCK_TOOL,
                    payload={"tools": sorted(_SEND_TOOLS), "reason": "User rule: never send emails automatically."},
                )
            ],
        )
        return RuleParseResult(
            rule=rule,
            explanation=f"I’ll block sending emails.",
            confidence=0.95,
            needs_confirmation=False,
        )

    # 2) "never forward emails" => block forward tool
    if mentions_forward and is_never:
        rule = Rule(
            id=_new_rule_id(),
            enabled=True,
            scope=scope,
            raw_text=raw,
            actions=[
                RuleAction(
                    type=RuleActionType.BLOCK_TOOL,
                    payload={"tools": sorted(_FORWARD_TOOLS), "reason": "User rule: never forward emails."},
                )
            ],
        )
        return RuleParseResult(
            rule=rule,
            explanation=f"I’ll block forwarding emails.",
            confidence=0.95,
            needs_confirmation=False,
        )

    # 3) "confirm before sending" / "always show draft first" => require confirmation on execute draft
    if mentions_send and (wants_confirm or wants_draft_first or is_always):
        # Only treat "always ..." as confirm-send if it clearly references draft/confirm or sending.
        if wants_confirm or wants_draft_first:
            reason = "User rule: require confirmation before sending emails."
            if wants_draft_first:
                reason = "User rule: show a draft and require confirmation before sending emails."
            rule = Rule(
                id=_new_rule_id(),
                enabled=True,
                scope=scope,
                raw_text=raw,
                actions=[
                    RuleAction(
                        type=RuleActionType.CONFIRM_TOOL,
                        payload={"tools": sorted(_SEND_TOOLS), "reason": reason},
                    )
                ],
            )
            return RuleParseResult(
                rule=rule,
                explanation=f"I’ll require confirmation before sending emails.",
                confidence=0.9,
                needs_confirmation=False,
            )

    # 4) "never delete drafts" => block delete draft
    if mentions_delete and is_never:
        rule = Rule(
            id=_new_rule_id(),
            enabled=True,
            scope=scope,
            raw_text=raw,
            actions=[
                RuleAction(
                    type=RuleActionType.BLOCK_TOOL,
                    payload={"tools": sorted(_DELETE_TOOLS), "reason": "User rule: never delete drafts."},
                )
            ],
        )
        return RuleParseResult(
            rule=rule,
            explanation=f"I’ll block deleting drafts.",
            confidence=0.9,
            needs_confirmation=False,
        )

    return None
