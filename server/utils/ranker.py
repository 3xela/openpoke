from sentence_transformers import SentenceTransformer
import numpy as np


class AgentRanker:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", top_k: int = 5):
        self.model = SentenceTransformer(model_name)
        self.top_k = top_k

        self.agent_names: list[str] = []
        self.embeddings: np.ndarray | None = None 

    def _dedupe_exact(self, agent_list: list[str]) -> list[str]:
        """Deduplicate exact names with basic normalization."""
        seen: set[str] = set()
        unique: list[str] = []

        for name in agent_list:
            if not isinstance(name, str):
                continue
            cleaned = " ".join(name.strip().split())
            if not cleaned:
                continue

            key = cleaned.lower()
            if key in seen:
                continue

            seen.add(key)
            unique.append(cleaned)

        return unique

    def encode_agents(self, agent_list: list[str]) -> None:
        """Store (deduped) agent names and compute embeddings."""
        unique_agents = self._dedupe_exact(agent_list)
        self.agent_names = unique_agents

        if not unique_agents:
            self.embeddings = None
            return

        self.embeddings = self.model.encode(unique_agents)

    def search_top_k(self, query: str) -> list[str]:
        """Return top_k most similar agent names to the query."""
        if self.embeddings is None or not self.agent_names:
            return []

        query_emb = self.model.encode(query)  # (D,)

        # cosine similarity, vectorized: (N,)
        scores = (self.embeddings @ query_emb) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb)
        )

        k = min(self.top_k, len(self.agent_names))
        top_indices = scores.argsort()[-k:][::-1]
        return [self.agent_names[i] for i in top_indices]
