from typing import Any, Dict
from pydantic import BaseModel, Field


class TransformResult(BaseModel):
    """result of a transformer call"""
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}
