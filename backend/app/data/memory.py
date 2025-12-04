from typing import List, Dict, Any
from dataclasses import dataclass, field
import hashlib


@dataclass
class ConversationMemory:
    """Lightweight per-agent memory with rolling buffer and simple summarization."""

    max_messages: int = 50
    messages: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        if not content:
            return
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self.max_messages:
            self.messages = compress_messages(self.messages, max_items=self.max_messages)

    def recent(self, k: int = 5) -> List[str]:
        return [m.get("content", "") for m in self.messages[-k:]]

    def summarize(self, k: int = 8, max_chars: int = 600) -> str:
        """Naive summary by concatenation of last k messages."""
        txt = "\n".join(self.recent(k))
        return txt[:max_chars]


def compress_messages(messages: List[Dict[str, Any]], max_items: int = 20) -> List[Dict[str, Any]]:
    """Compress long histories by keeping head/tail and hashing dropped content."""
    if len(messages) <= max_items:
        return messages
    keep_head = messages[: max_items // 3]
    keep_tail = messages[-(max_items - len(keep_head)) :]
    dropped = messages[len(keep_head) : -len(keep_tail)]
    digest = hashlib.md5("".join(d.get("content", "") for d in dropped).encode("utf-8")).hexdigest()
    marker = {"role": "system", "content": f"[compressed {len(dropped)} msgs | md5={digest}]"}
    return keep_head + [marker] + keep_tail
