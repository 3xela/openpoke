"""Interaction agent helpers for prompt construction."""
import re
from html import escape
from pathlib import Path
from typing import Dict, List

from ...services.execution import get_agent_roster
from ...services.memory import Memory
from ...utils import AgentRanker

_prompt_path = Path(__file__).parent / "system_prompt.md"
SYSTEM_PROMPT = _prompt_path.read_text(encoding="utf-8").strip()


USER_MSG_RE = re.compile(r"<user_message\b[^>]*>(.*?)</user_message>", re.DOTALL)

def user_only_from_transcript(transcript: str) -> str:
    parts = [m.strip() for m in USER_MSG_RE.findall(transcript)]
    return "\n".join(p for p in parts if p)

# Load and return the pre-defined system prompt from markdown file
def build_system_prompt() -> str:
    """Return the static system prompt for the interaction agent."""
    return SYSTEM_PROMPT


# Build structured message with conversation history, active agents, memory and current turn 
def prepare_message_with_memory(
    latest_text: str,
    transcript: str,
    ranker: AgentRanker,
    memory: Memory,
    message_type: str = "user",
) -> List[Dict[str, str]]:
    sections: List[str] = []

    user_transcript = user_only_from_transcript(transcript)

    if message_type == "user":
        memory.semantic_compression(latest_text, baseline_text=user_transcript)

    sections.append(_render_conversation_history(transcript))
    sections.append(f"<active_agents>\n{_render_relevant_agents(latest_text, ranker)}\n</active_agents>")
    sections.append(_render_current_turn(latest_text, message_type))
    sections.append(f"<memories>\n{memory.get_text_memories_str()}\n</memories>")

    content = "\n\n".join(sections)
    return [{"role": "user", "content": content}]

# Build structured message with conversation history, active agents, and current turn
def prepare_message_with_history(
    latest_text: str,
    transcript: str,
    ranker: AgentRanker,
    message_type: str = "user",
) -> List[Dict[str, str]]:
    """Compose a message that bundles history, roster, and the latest turn."""
    sections: List[str] = []

    sections.append(_render_conversation_history(transcript))
    sections.append(f"<active_agents>\n{_render_relevant_agents(latest_text, ranker)}\n</active_agents>")
    sections.append(_render_current_turn(latest_text, message_type))

    content = "\n\n".join(sections)
    return [{"role": "user", "content": content}]


# Format conversation transcript into XML tags for LLM context
def _render_conversation_history(transcript: str) -> str:
    history = transcript.strip()
    if not history:
        history = "None"
    return f"<conversation_history>\n{history}\n</conversation_history>"


# Format currently active execution agents into XML tags for LLM awareness
def _render_active_agents() -> str:
    roster = get_agent_roster()
    roster.load()
    agents = roster.get_agents()

    if not agents:
        return "None"

    rendered: List[str] = []
    for agent_name in agents:
        name = escape(agent_name or "agent", quote=True)
        rendered.append(f'<agent name="{name}" />')

    return "\n".join(rendered)


def _render_relevant_agents(latest_text: str , ranker: AgentRanker)-> str:
    roster = get_agent_roster()
    roster.load()
    agents = roster.get_agents()

    if not agents:
        return "None"
    
    ranker.encode_agents(agents)
    ranked_agents = ranker.search_top_k(latest_text)
    rendered: List[str] = []
    for agent_name in ranked_agents:
        name = escape(agent_name or "agent", quote=True)
        rendered.append(f'<agent name="{name}" />')

    return "\n".join(rendered)

# Wrap the current message in appropriate XML tags based on sender type
def _render_current_turn(latest_text: str, message_type: str) -> str:
    tag = "new_agent_message" if message_type == "agent" else "new_user_message"
    body = latest_text.strip()
    return f"<{tag}>\n{body}\n</{tag}>"
