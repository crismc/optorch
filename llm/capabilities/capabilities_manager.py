from typing import Any, Dict, List, Optional
from optorch.llm.capabilities.capability import Capability
from optorch.llm.capabilities.capabilities_registry import CapabilitiesRegistry

class LLMCapabilitiesManager:    
    def __init__(self, registry: CapabilitiesRegistry) -> None:
        self.registry = registry
        self._capabilities: Dict[str, Capability] = {}
        self._profile_activation: Dict[str, List[str]] = {}
    
    def register_profile(self, name: str, capabilities: List[str]) -> None:
        if capabilities:
            self._profile_activation[name] = list(capabilities)
    
    def get_active(self, profile_name: str) -> List[str]:
        return self._profile_activation.get(profile_name, [])
    
    def _get(self, provider: str, name: str) -> Optional[Capability]:
        key = f"{provider}:{name}"
        cached = self._capabilities.get(key)

        if cached is not None:
            return cached
        
        cls = self.registry.get_for(provider, name)

        if cls is None:
            return None
        
        event_type = self.registry.event_type_for(name) or f"llm.capability.{name}"
        params = self.registry.get_params_for(provider, name)
        instance = cls(event_type=event_type, **params)
        self._capabilities[key] = instance

        return instance
    
    def _model_matches(self, provider: str, name: str, model: str) -> bool:
        patterns = self.registry.get_patterns_for(provider, name)

        if not patterns:
            return True
        
        return any(p in model for p in patterns)
    
    def produce(self, provider: str, model: str, active: List[str]) -> Dict[str, Any]:
        params: Dict[str, Any] = {}

        for cap_name in active:
            if not self._model_matches(provider, cap_name, model):
                continue

            cap = self._get(provider, cap_name)

            if cap is None:
                continue

            contribution = cap.contribute(model)

            if contribution:
                params.update(contribution)

        return params
    
    def extract(self, provider: str, chunk: Any, active: List[str]) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []

        for cap_name in active:
            cap = self._get(provider, cap_name)

            if cap is None:
                continue

            event = cap.extract(chunk)

            if event:
                if cap.accumulation_key:
                    event["_accumulation_key"] = cap.accumulation_key
                events.append(event)

        return events
