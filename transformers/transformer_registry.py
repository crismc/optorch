from typing import Type, TYPE_CHECKING
from optorch.registry import Registry
from optorch.transformers.base_transformer import BaseTransformer
from optorch.transformers.transform_result import TransformResult

if TYPE_CHECKING:
    from optorch.llm.lifecycle.context import LLMContext


class TransformerRegistry:
    """Registry for response transformers"""
    
    def __init__(self) -> None:
        self._registry = Registry[Type[BaseTransformer]]()
    
    def register(self, name: str, transformer: Type[BaseTransformer]):
        """Register a transformer class by name"""
        self._registry.register(name, transformer)
    
    def get(self, name: str) -> BaseTransformer:
        """Get transformer instance by name"""
        transformer_class = self._registry.get(name)
        return transformer_class()
    
    def has(self, name: str) -> bool:
        """Check if transformer exists"""
        return self._registry.has(name)
    
    async def apply(self, name: str, content: str, context: 'LLMContext') -> TransformResult:
        """Apply a transformer by name. Returns TransformResult."""
        transformer = self.get(name)
        return await transformer.transform(content, context)
