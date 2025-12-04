from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union


@dataclass
class LLMMessage:
    role: str
    content: Union[str, Dict[str, Any]]
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        msg: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            msg["name"] = self.name
        return msg


def to_openrouter_messages(messages: List[Union["LLMMessage", Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Normalize a list of messages (LLMMessage or already-formed dicts) to OpenRouter format."""
    out: List[Dict[str, Any]] = []
    for m in messages:
        if isinstance(m, LLMMessage):
            out.append(m.to_dict())
        elif isinstance(m, dict) and "role" in m and "content" in m:
            out.append(dict(m))
    return out
