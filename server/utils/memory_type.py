from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MemoryItem:
    text: str
    embedding: Optional[List[float]] = None
    sim_score : float # this is for debugging, we need to know what the sim score is when we create a memoryitem 


@dataclass
class LLMMemory:
    memories: List[MemoryItem] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "memories": [
                {"text": m.text, "embedding": m.embedding}
                for m in self.memories
            ]
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LLMMemory":
        items = []
        for raw in data.get("memories", []):
            items.append(
                MemoryItem(
                    text=raw["text"],
                    embedding=raw.get("embedding"),
                )
            )
        return cls(memories=items)
