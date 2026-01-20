# server/agents/interaction_agent/deps.py
from ..services.rules import RuleStore
from .ranker import AgentRanker
from sentence_transformers import SentenceTransformer
from .memory import Memory


_ranker = None
_rule_store = None
_embedder = None
_memory = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder

def get_agent_ranker():
    global _ranker
    if _ranker is None:
        _ranker = AgentRanker(model = get_embedder())
    return _ranker

def get_rule_store():
    global _rule_store
    if _rule_store is None:
        _rule_store = RuleStore()
    return _rule_store

def get_memory():
    global _memory
    if _memory is None:
        _memory = Memory(embedder=get_embedder())
    return _memory
