from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Set
import numpy as np
import re

from sentence_transformers import SentenceTransformer

from .memory_type import LLMMemory, MemoryItem



WORD_RE = re.compile(r"[a-zA-Z0-9_]+")

STOPWORDS = {
    "the","a","an","and","or","but","if","then","else","so","to","of","in","on","for","with","at","by",
    "is","are","was","were","be","been","being","it","this","that","these","those",
    "i","you","he","she","we","they","me","my","your","our","their",
}

def unique_words(text: str) -> Set[str]:
    words = {w.lower() for w in WORD_RE.findall(text)}
    words = {w for w in words if len(w) >= 3 and w not in STOPWORDS}
    return words

#use a unique word counter for E_new, also ignore very common low information words from stopwords. 
# ideally we would need a way to check for "high information words" with an embedding model, and compare the ratio

def new_word_ratio(window_text: str, prev_text: str) -> float:
    w = unique_words(window_text)
    if not w:
        return 0.0
    prev = unique_words(prev_text)
    new = w - prev
    return len(new) / len(w)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


class Memory:
    def __init__(self, embedder: SentenceTransformer):
        self.model = embedder
        self.memory_path = Path(__file__).resolve().parents[2] / "data" / "memory.json"
        self.memory: LLMMemory = self.load_memories()
        self.tau = 0.75
        self.alpha = 0.5
        self.window_size = 255 #255 chars of sliding window , this is arbitrary rn, needs some testing + thought but this will grab a couple words.

    def load_memories(self) -> LLMMemory:
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)

        if self.memory_path.exists():
            with self.memory_path.open("r", encoding="utf-8") as f:
                return LLMMemory.from_dict(json.load(f))

        mem = LLMMemory()
        self._save(mem)
        return mem

    def _save(self, mem: Optional[LLMMemory] = None) -> None:
        if mem is None:
            mem = self.memory
        with self.memory_path.open("w", encoding="utf-8") as f:
            json.dump(mem.to_dict(), f, indent=2)

    def create_memory(self, memory_text_window: str) -> MemoryItem:
        emb = self.model.encode(memory_text_window)
        item = MemoryItem(
            text=memory_text_window,
            embedding=emb.tolist(), 
        )
        self.memory.memories.append(item)
        self._save()
        return item

    def information_score(
        self,
        window: str,
        prev_text: str,
        prev_embed: np.ndarray,
    ) -> float:
        window_embed = self.model.encode(window)
        sim = cosine_sim(window_embed, prev_embed) 
        novelty = new_word_ratio(window, prev_text)
        score = self.alpha * novelty + (1 - self.alpha) * (1 - sim)
        return float(score)

    def semantic_compression(self, history: str) -> None:

        history_embed = self.model.encode(history)

        n = len(history)
        end = max(1, n - self.window_size + 1)

        for i in range(0, end, self.step):
            window = history[i : i + self.window_size]
            prev_text = history[:i]
            score = self.information_score(window, prev_text, history_embed)

            if score >= self.tau:
                self.create_memory(window)

    def get_text_memories_str(self) -> str:
        return "\n".join(m.text for m in self.memory.memories if m.text)

        