"""
OpenAI LLM client implementation.
"""
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from optorch.llm.base_client import BaseLLMClient
from optorch.llm.responses import LLMResponseFactory, StandardLLMResponse, StreamingLLMResponse
from optorch.llm.responses.helpers.openai_extractor import OpenAIExtractor
from optorch.llm.metrics import Usage
from optorch.events import emits, EventTypes
from optorch.filters import FilterManager

if TYPE_CHECKING:
    from optorch.llm.lifecycle.context import LLMContext

class OpenAIClient(BaseLLMClient):
    """OpenAI API client (GPT-4, GPT-4o, etc.)"""

    MODEL_PATTERNS = ["gpt-", "o1-", "text-embedding", "davinci", "curie", "babbage", "ada"]

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        tpm_limit: int = 90000
    ):
        super().__init__(model=model, tpm_limit=tpm_limit)
        self.api_key = api_key
        self.temperature = temperature
        self._client = None
    
    @property
    def client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai package not installed. Install with: pip install openai")
            
        return self._client
    
    @emits(EventTypes.LLM)
    async def invoke(self, context: 'LLMContext', messages: List[Dict[str, Any]], **kwargs) -> StandardLLMResponse:
        self.active_requests += 1

        try:
            temperature = kwargs.pop("temperature", self.temperature)
            tools = kwargs.pop("tools", None)
            filtered = FilterManager.for_target("messages", "openai").apply(messages)
            params = await self._build_invoke_params(filtered, temperature, tools, **kwargs)
            response = await self._call_api(params)

            return LLMResponseFactory.create(
                content=self._extract_content(response),
                tool_calls=self._extract_tool_calls(response),
                usage=self._extract_usage(response),
                raw_response=response,
            )
        finally:
            self.active_requests -= 1

    @emits(EventTypes.LLM)
    async def astream(self, context: 'LLMContext', messages: List[Dict[str, Any]], **kwargs) -> StreamingLLMResponse:
        self.active_requests += 1

        try:
            temperature = kwargs.pop("temperature", self.temperature)
            tools = kwargs.pop("tools", None)
            budget = kwargs.pop("budget", None)
            completion_type = kwargs.pop("completion_type", "hard_stop")
            filtered = FilterManager.for_target("messages", "openai").apply(messages)
            params = await self._build_stream_params(filtered, temperature, tools, **kwargs)
            stream = await self._call_stream_api(params)

            return StreamingLLMResponse(
                stream=stream,
                model=self.model,
                provider="openai",
                metadata={"temperature": temperature, "tools": tools is not None},
                budget=budget,
                completion_type=completion_type,
            )
        finally:
            self.active_requests -= 1

    async def _build_invoke_params(self, messages: List[Dict[str, Any]], temperature: float, tools: Optional[List], **kwargs) -> Dict[str, Any]:
        if OpenAIExtractor.uses_responses_api(self.model):
            input_messages = OpenAIExtractor.convert_messages(messages)
            params: Dict[str, Any] = {"model": self.model, "input": input_messages, **kwargs}

            if tools:
                params["tools"] = OpenAIExtractor.adapt_tools(tools)

            return params

        params = {
            "model": self.model, 
            "messages": messages, 
            "temperature": temperature, 
            **kwargs
        }

        if tools:
            params["tools"] = tools

        return params

    async def _build_stream_params(self, messages: List[Dict[str, Any]], temperature: float, tools: Optional[List], **kwargs) -> Dict[str, Any]:
        params = await self._build_invoke_params(messages, temperature, tools, **kwargs)
        params["stream"] = True

        if "messages" in params:
            params["stream_options"] = {"include_usage": True}

        return params

    async def _call_api(self, params: Dict[str, Any]) -> Any:
        if "input" in params:
            return await self.client.responses.create(**params)
        
        return await self.client.chat.completions.create(**params)

    async def _call_stream_api(self, params: Dict[str, Any]) -> Any:
        if "input" in params:
            return await self.client.responses.create(**params)
        
        return await self.client.chat.completions.create(**params)

    def _extract_content(self, response: Any) -> str:
        if hasattr(response, "output_text"):
            return response.output_text or ""
        
        return response.choices[0].message.content or ""

    def _extract_tool_calls(self, response: Any) -> Optional[List[Dict[str, Any]]]:
        if hasattr(response, "output"):
            calls = [
                {"id": item.call_id, "type": "function", "function": {"name": item.name, "arguments": item.arguments}}
                for item in response.output
                if getattr(item, "type", None) == "function_call"
            ]

            return calls or None

        if not response.choices[0].message.tool_calls:
            return None
        
        return [
            {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in response.choices[0].message.tool_calls
        ]

    def _extract_usage(self, response: Any) -> Optional[Usage]:
        model_key = f"{self.provider_prefix}/{self.model}" if self.provider_prefix else (self.model or "unknown")
        from optorch.llm.pricing import Pricing

        if hasattr(response, "output_text"):
            usage = getattr(response, "usage", None)

            if not usage:
                return None
            
            return Usage.create(model_key, usage.input_tokens, usage.output_tokens, currency=Pricing.get_currency())

        usage_data = response.usage

        if not usage_data:
            return None
        
        return Usage.create(model_key, usage_data.prompt_tokens, usage_data.completion_tokens, currency=Pricing.get_currency())

    def _get_provider_name(self) -> str:
        return "openai"
