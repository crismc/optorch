from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from optorch.transformers.transform_result import TransformResult

if TYPE_CHECKING:
    from optorch.llm.lifecycle.context import LLMContext


class BaseTransformer(ABC):
    """Base class for content transformers that clean/normalize LLM response text"""
    
    @abstractmethod
    async def transform(self, content: str, context: 'LLMContext') -> TransformResult:
        """Transform content string.
        
        Args:
            content: The response content string
            context: Full LLM context for state/events/metadata access
            
        Returns:
            TransformResult with transformed content and optional extracted metadata
        """
        pass
