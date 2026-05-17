from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, Optional


class Capability(ABC):    
    provider: ClassVar[str]
    name: ClassVar[str]
    accumulation_key: ClassVar[Optional[str]] = None
    
    def __init__(self, event_type: str) -> None:
        self.event_type = event_type
    
    @abstractmethod
    def contribute(self, model: str) -> Optional[Dict[str, Any]]:
        ...
    
    @abstractmethod
    def extract(self, chunk: Any) -> Optional[Dict[str, Any]]:
        ...
