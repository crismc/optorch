from typing import List
from pydantic import BaseModel, ConfigDict

from optorch.llm.capabilities.capabilities_manager import LLMCapabilitiesManager


class CapabilityContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    active: List[str]
    manager: LLMCapabilitiesManager
