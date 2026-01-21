from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MemoryItem:
    text: str
    sim_score : float = 0.0 # this is for debugging, we need to know what the sim score is when we create a memoryitem 



@dataclass
class LLMMemory:
    memories: List[MemoryItem] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "memories": [
                {"text": m.text, "sim_score": m.sim_score}
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
                    sim_score=raw["sim_score"],
                )
            )
        return cls(memories=items)
