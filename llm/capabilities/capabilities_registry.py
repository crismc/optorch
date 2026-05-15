from importlib import import_module
from typing import List, Optional, Set

from optorch.registry import Registry
from optorch.llm.capabilities.capability import Capability
from optorch.llm.capabilities.config import CapabilitiesConfig


class CapabilitiesRegistry(Registry[type[Capability]]):    
    def __init__(self) -> None:
        super().__init__()
        self._event_types: dict[str, str] = {}
        self._model_patterns: dict[str, List[str]] = {}
        self._params: dict[str, dict[str, object]] = {}
    
    @classmethod
    def from_config(cls, config: CapabilitiesConfig) -> "CapabilitiesRegistry":
        registry = cls()

        for cap in config.capabilities:
            registry._event_types[cap.name] = cap.event_type

            for binding in cap.providers:
                module_path, _, class_name = binding.class_.rpartition(".")
                module = import_module(module_path)
                cap_class: type[Capability] = getattr(module, class_name)
                key = f"{binding.provider}:{cap.name}"
                registry.register(key, cap_class)
                registry._model_patterns[key] = list(binding.model_patterns)
                registry._params[key] = dict(binding.params)

        return registry
    
    def get_for(self, provider: str, name: str) -> Optional[type[Capability]]:
        return self.get_optional(f"{provider}:{name}")
    
    def get_patterns_for(self, provider: str, name: str) -> List[str]:
        return self._model_patterns.get(f"{provider}:{name}", [])

    def get_params_for(self, provider: str, name: str) -> dict[str, object]:
        return self._params.get(f"{provider}:{name}", {})
    
    def event_type_for(self, name: str) -> Optional[str]:
        return self._event_types.get(name)
    
    def known_names(self) -> Set[str]:
        return set(self._event_types.keys())
