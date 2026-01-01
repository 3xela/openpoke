# server/services/rules/store.py
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import Rule, RuleScope  # assumes these exist in models.py


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(o: Any) -> Any:
    """
    JSON serializer fallback:
    - dataclasses -> dict
    - enums -> value
    """
    if is_dataclass(o):
        return asdict(o)
    # Enum-like (RuleScope / RuleActionType, etc.)
    if hasattr(o, "value"):
        return o.value
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _atomic_write_text(path: Path, text: str) -> None:
    """
    Atomic write (best-effort cross-platform): write to temp file in same dir then replace.
    """
    _ensure_parent_dir(path)
    dir_path = str(path.parent)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=dir_path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        # If replace failed, clean up tmp
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass


def _coerce_scope(scope: Any) -> RuleScope:
    if isinstance(scope, RuleScope):
        return scope
    # allow strings like "email", "chat", "global"
    return RuleScope(str(scope))


def _rule_to_dict(rule: Rule) -> Dict[str, Any]:
    """
    Convert Rule -> JSON dict. Assumes Rule/actions are dataclasses/enums.
    """
    data = asdict(rule) if is_dataclass(rule) else dict(rule)  # type: ignore[arg-type]
    # ensure enum serialization
    if "scope" in data and hasattr(data["scope"], "value"):
        data["scope"] = data["scope"].value
    return data


def _rule_from_dict(d: Dict[str, Any]) -> Rule:
    """
    Convert JSON dict -> Rule.

    This is intentionally forgiving: extra keys are ignored if your Rule dataclass
    doesn't accept them; missing optional keys get defaults via Rule constructor.
    """
    # Normalize scope if present
    if "scope" in d:
        d = dict(d)
        d["scope"] = _coerce_scope(d["scope"])

    # Rule constructor must accept at least: id, enabled, scope, raw_text, actions
    # If your Rule model differs, adjust here.
    return Rule(**d)  # type: ignore[arg-type]

def _default_rules_path() -> Path:
    # store.py lives at: server/services/rules/store.py
    # parents[3] -> server/
    server_root = Path(__file__).resolve().parents[2]
    return server_root / "data" / "rules.json"

class RuleStore:
    """
    JSON-backed rule store.

    - Keeps in-memory cache.
    - Saves atomically on mutations.
    - Minimal CRUD: list/add/enable-disable/delete.
    """
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else _default_rules_path()
        self._rules: Dict[str, Rule] = {}

        if not self.path.exists():
            self.save()   # creates server/data/ and a valid empty rules.json
        else:
            self.load()

    def load(self) -> None:
        if not self.path.exists():
            self._rules = {}
            return

        raw = self.path.read_text(encoding="utf-8").strip()
        if not raw:
            self._rules = {}
            return

        data = json.loads(raw)
        # Accept either {"rules":[...]} or a bare list [...]
        rules_list = data.get("rules") if isinstance(data, dict) else data
        if rules_list is None:
            self._rules = {}
            return

        out: Dict[str, Rule] = {}
        for item in rules_list:
            if not isinstance(item, dict):
                continue
            try:
                rule = _rule_from_dict(item)
                out[rule.id] = rule
            except Exception:
                # Skip malformed rule entries rather than crashing startup
                continue
        self._rules = out

    def save(self) -> None:
        payload = {"version": 1, "updated_at": _utc_now_iso(), "rules": [_rule_to_dict(r) for r in self._rules.values()]}
        text = json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
        _atomic_write_text(self.path, text + "\n")

    def list_rules(self, scope: Optional[RuleScope] = None, enabled_only: bool = False) -> List[Rule]:
        rules = list(self._rules.values())
        if scope is not None:
            scope = _coerce_scope(scope)
            rules = [r for r in rules if r.scope == scope]
        if enabled_only:
            rules = [r for r in rules if getattr(r, "enabled", True)]
        # Stable ordering: by id
        rules.sort(key=lambda r: r.id)
        return rules

    def get_rule(self, rule_id: str) -> Optional[Rule]:
        return self._rules.get(rule_id)

    def add_rule(self, rule: Rule, save: bool = True) -> Rule:
        # Ensure id exists
        if not getattr(rule, "id", None):
            raise ValueError("Rule.id must be set before adding to store")

        # Optional timestamps if your Rule model has them
        if hasattr(rule, "created_at") and getattr(rule, "created_at", None) in (None, ""):
            try:
                setattr(rule, "created_at", _utc_now_iso())
            except Exception:
                pass
        if hasattr(rule, "updated_at"):
            try:
                setattr(rule, "updated_at", _utc_now_iso())
            except Exception:
                pass

        self._rules[rule.id] = rule
        if save:
            self.save()
        return rule

    def set_enabled(self, rule_id: str, enabled: bool, save: bool = True) -> bool:
        rule = self._rules.get(rule_id)
        if rule is None:
            return False
        if hasattr(rule, "enabled"):
            setattr(rule, "enabled", bool(enabled))
        else:
            # If your Rule model lacks enabled, treat as non-toggleable
            return False

        if hasattr(rule, "updated_at"):
            try:
                setattr(rule, "updated_at", _utc_now_iso())
            except Exception:
                pass

        if save:
            self.save()
        return True

    def delete(self, rule_id: str, save: bool = True) -> bool:
        if rule_id not in self._rules:
            return False
        del self._rules[rule_id]
        if save:
            self.save()
        return True
