from typing import Any, Dict, List
from pydantic import BaseModel, Field

class CapabilityProviderConfig(BaseModel):
    provider: str = Field(description="provider name (openai, anthropic, ollama, ...)")
    class_: str = Field(alias="class", description="dotted import path of the Capability subclass")
    model_patterns: List[str] = Field(default_factory=list, description="model substring patterns that activate this capability - empty = all models")
    params: Dict[str, Any] = Field(default_factory=dict, description="extra kwargs forwarded to the capability constructor")
    
    model_config = {"populate_by_name": True}

class CapabilityConfig(BaseModel):
    name: str = Field(description="capability name e.g. reasoning")
    event_type: str = Field(description="event type emitted to consumers")
    providers: List[CapabilityProviderConfig] = Field(default_factory=list)

def _default_capabilities() -> List[CapabilityConfig]:
    return [
        CapabilityConfig(
            name="reasoning",
            event_type="llm.capability.reasoning",
            providers=[
                CapabilityProviderConfig.model_validate({
                    "provider": "openai",
                    "class": "optorch.llm.capabilities.openai.openai_reasoning.OpenAIReasoning",
                    "model_patterns": ["o1", "o3", "o4", "gpt-5"],
                    "params": {"effort": "medium"},
                }),
                CapabilityProviderConfig.model_validate({
                    "provider": "anthropic",
                    "class": "optorch.llm.capabilities.anthropic.anthropic_reasoning.AnthropicReasoning",
                    "model_patterns": ["claude-opus-4", "claude-sonnet-4", "claude-3-7", "claude-3.7"],
                    "params": {"budget_tokens": 8000},
                }),
                CapabilityProviderConfig.model_validate({
                    "provider": "ollama",
                    "class": "optorch.llm.capabilities.ollama.ollama_reasoning.OllamaReasoning",
                    "model_patterns": ["deepseek-r1", "qwen3", "qwq"],
                }),
            ],
        ),
    ]

class CapabilitiesConfig(BaseModel):
    capabilities: List[CapabilityConfig] = Field(default_factory=_default_capabilities)
