"""openai/groq chunk extractor"""
import json
from typing import Any, Optional, Dict, List, cast
from optorch.llm.metrics import Usage


class OpenAIExtractor:
    _RESPONSES_API_MODELS = ("o1", "o3", "o4", "gpt-5")

    @classmethod
    def uses_responses_api(cls, model: Optional[str]) -> bool:
        """OpenAI is a pain!!! o-series / gpt-5 stream through the Responses API, not Chat Completions"""
        return bool(model and model.startswith(cls._RESPONSES_API_MODELS))

    @staticmethod
    def _is_responses_event(chunk: Any) -> bool:
        t = getattr(chunk, "type", None)
        return isinstance(t, str) and t.startswith("response.")

    @staticmethod
    def extract_content(chunk: Any) -> Optional[str]:
        if OpenAIExtractor._is_responses_event(chunk):
            if getattr(chunk, "type") == "response.output_text.delta":
                return getattr(chunk, "delta", None)
            
            return None
        
        if hasattr(chunk, "choices") and chunk.choices:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                return delta.content
            
        return None

    @staticmethod
    def extract_tool_calls(chunk: Any) -> Optional[Any]:
        if OpenAIExtractor._is_responses_event(chunk):
            if getattr(chunk, "type") != "response.output_item.done":
                return None
            
            item = getattr(chunk, "item", None)
            if item and getattr(item, "type", None) == "function_call":
                return {
                    "id": getattr(item, "call_id", None) or getattr(item, "id", None),
                    "name": getattr(item, "name", None),
                    "arguments": getattr(item, "arguments", "{}"),
                    "__responses_format": True,
                }
            
            return None
        
        if hasattr(chunk, "choices") and chunk.choices:
            delta = chunk.choices[0].delta
            if hasattr(delta, "tool_calls") and delta.tool_calls:
                return delta.tool_calls
        return None

    @staticmethod
    def extract_usage(chunk: Any, model: str) -> Optional[Usage]:
        if OpenAIExtractor._is_responses_event(chunk):
            if getattr(chunk, "type") != "response.completed":
                return None
            
            response = getattr(chunk, "response", None)
            usage = getattr(response, "usage", None) if response else None

            if usage:
                return Usage.create(model, getattr(usage, "input_tokens", 0), getattr(usage, "output_tokens", 0))
            
            return None
        
        if hasattr(chunk, "usage") and chunk.usage:
            return Usage.create(model, chunk.usage.prompt_tokens, chunk.usage.completion_tokens)
        
        return None

    @staticmethod
    def create_tool_buffer() -> Dict[int, Dict[str, Any]]:
        return {}

    @staticmethod
    def adapt_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        result = []

        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                fn = tool["function"]
                result.append({
                    "type": "function",
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                })
            else:
                result.append(tool)

        return result

    @staticmethod
    def convert_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        _ALLOWED_MSG_KEYS = {"role", "content", "name"}
        converted: List[Dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role")

            if role == "tool":
                converted.append({
                    "type": "function_call_output",
                    "call_id": msg["tool_call_id"],
                    "output": msg.get("content") or "",
                })
                
                continue

            if role == "assistant" and msg.get("tool_calls"):
                content = msg.get("content")

                if content:
                    converted.append({"role": "assistant", "content": content})

                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    converted.append({
                        "type": "function_call",
                        "call_id": tc["id"],
                        "name": fn.get("name", ""),
                        "arguments": fn.get("arguments", "{}"),
                    })

                continue

            entry: Dict[str, Any] = {k: v for k, v in msg.items() if k in _ALLOWED_MSG_KEYS}

            if entry.get("content") is None:
                entry["content"] = ""

            converted.append(entry)

        return converted

    @staticmethod
    def accumulate_tools(tool_calls: Any, buffer: Dict[int, Dict[str, Any]]) -> None:
        if isinstance(tool_calls, dict) and tool_calls.get("__responses_format"):
            idx = len(buffer)
            buffer[idx] = {
                "id": tool_calls["id"],
                "type": "function",
                "function": {"name": tool_calls["name"], "arguments": tool_calls["arguments"]},
            }

            return
        
        from optorch.llm.responses.helpers import accumulate_tool_calls
        accumulate_tool_calls(cast(List[Any], tool_calls), buffer)

    @staticmethod
    def finalize_tools(buffer: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        calls = []

        for idx in sorted(buffer.keys()):
            tool = buffer[idx]
            args = tool["function"]["arguments"]

            if isinstance(args, str):
                try:
                    tool["function"]["arguments"] = json.loads(args)
                except json.JSONDecodeError:
                    pass

            calls.append(tool)

        return calls
