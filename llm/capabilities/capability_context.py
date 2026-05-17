from typing import Any, Dict, List

from optorch.llm.capabilities.capabilities_manager import LLMCapabilitiesManager


class CapabilityContext:

    def __init__(self, active: List[str], manager: LLMCapabilitiesManager) -> None:
        self.active = active
        self.manager = manager
        self.accumulated: Dict[str, str] = {}

    def extract(self, provider: str, chunk: Any) -> List[Dict[str, Any]]:
        events = self.manager.extract(provider, chunk, self.active)

        for event in events:
            key = event.pop("_accumulation_key", None)
            if key:
                text = event.get("content", "")
                if text:
                    self.accumulated[key] = self.accumulated.get(key, "") + text

        return events
