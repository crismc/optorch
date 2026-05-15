from typing import Any, ClassVar, Dict, Optional
from optorch.llm.capabilities.capability import Capability

class AnthropicReasoning(Capability):   
    provider: ClassVar[str] = "anthropic"
    name: ClassVar[str] = "reasoning"
    
    DEFAULT_BUDGET_TOKENS: ClassVar[int] = 1024

    def __init__(self, event_type: str, budget_tokens: Optional[int] = None) -> None:
        super().__init__(event_type)
        self.budget_tokens = budget_tokens or self.DEFAULT_BUDGET_TOKENS

    def contribute(self, model: str) -> Optional[Dict[str, Any]]:
        return {"thinking": {"type": "enabled", "budget_tokens": self.budget_tokens}, "temperature": 1}
    
    def extract(self, chunk: Any) -> Optional[Dict[str, Any]]:
        delta = getattr(chunk, "delta", None)

        if delta is None:
            return None
        
        if getattr(delta, "type", None) != "thinking_delta":
            return None
        
        content = getattr(delta, "thinking", None)

        if not content:
            return None
        
        return {"type": self.event_type, "content": content}
