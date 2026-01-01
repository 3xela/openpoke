# server/agents/interaction_agent/deps.py
from ..services.rules.store import RuleStore
from .ranker import AgentRanker

_ranker = None
_rule_store = None

def get_agent_ranker():
    global _ranker
    if _ranker is None:
        _ranker = AgentRanker()
    return _ranker

def get_rule_store():
    global _rule_store
    if _rule_store is None:
        _rule_store = RuleStore()
    return _rule_store
