from optorch.logging import get_logger
from typing import Dict, Any, List, Optional, TYPE_CHECKING

from optorch.llm.lifecycle.base_processor import BaseLLMProcessor
from optorch.llm.lifecycle.hooks import LLMLifecycleHook
from optorch.llm.lifecycle.context import LLMContext
from optorch.messaging import MessageContext, Message
from optorch.events.event_types import EventTypes

if TYPE_CHECKING:
    from optorch.history.manager import History

logger = get_logger(__name__)

class HistoryPersistence(BaseLLMProcessor):

    def __init__(self):
        super().__init__()
        self.substates = {"default"}
        self.exclude_substates = {"tool_result"}

    @property
    def hook(self) -> LLMLifecycleHook:
        return LLMLifecycleHook.FINALIZE

    async def process(self, context: LLMContext) -> None:
        if not context.node_context or not context.node_context.history:
            return

        history: 'History' = context.node_context.history
        session_id = context.processor_data.get("session_id", "default")
        node_name = context.node_context.current_node_name
        messages: List[Message] = []
        event_messages: List[Dict[str, Any]] = []

        if context.messages:
            user_msg = context.messages[-1]
            user = Message(
                role=user_msg.get("role", "user"),
                content=user_msg.get("content", ""),
                metadata={}
            )
            messages.append(user)
            event_messages.append({
                "id": user.id,
                "role": user.role,
                "content": user.content,
                "model": None,
                "input_tokens": None,
                "output_tokens": None,
                "cost": None,
                "node_name": node_name,
                "capabilities": None,
                "metadata": None,
            })

        if context.response and context.response.content:
            caps = getattr(context.response, "capabilities", None)
            accumulated = caps.accumulated if caps else {}
            usage = context.metadata.get("usage")
            model: Optional[str] = getattr(context.response, "model", None)

            assistant_metadata: Dict[str, Any] = {}
            if context.response.tool_calls:
                assistant_metadata["tool_calls"] = context.response.tool_calls
            assistant_metadata.update(accumulated)

            assistant = Message(
                role="assistant",
                content=context.response.content,
                metadata=assistant_metadata
            )
            messages.append(assistant)

            event_messages.append({
                "id": assistant.id,
                "role": assistant.role,
                "content": assistant.content,
                "model": model,
                "input_tokens": usage.input_tokens if usage else None,
                "output_tokens": usage.output_tokens if usage else None,
                "cost": float(usage.cost) if usage else None,
                "node_name": node_name,
                "capabilities": accumulated or None,
                "metadata": {"tool_calls": context.response.tool_calls} if context.response.tool_calls else None,
            })

        if not messages:
            logger.warning("no messages to persist")
            return

        await history.save(messages, MessageContext(session_id=session_id))

        usage = context.metadata.get("usage")
        usage_dict = usage.to_dict() if usage else None
        model = getattr(context.response, "model", None) if context.response else None

        context.events.emit(
            f"{EventTypes.SESSION}.updated",
            {
                "session_id": session_id,
                "messages": event_messages,
                "usage": usage_dict,
                "model": model,
                "source": context.state.get("source") if context.state else None,
                "llm_context": {"messages": context.messages} if context.messages else None,
            },
            context.state
        )
