import asyncio
import json
from typing import Dict, Any, TYPE_CHECKING

from optorch.events.listeners.base import BaseListener
from optorch.logging import get_logger

if TYPE_CHECKING:
    from optorch.storage.manager import StorageManager
    from optorch.config import ConfigManager
    from optorch.session.session_manager import SessionManager

logger = get_logger(__name__)


class ConversationListener(BaseListener):

    def __init__(
        self,
        storage_manager: "StorageManager",
        config_manager: "ConfigManager",
        session_manager: "SessionManager | None" = None,
    ):
        super().__init__()
        self._storage_manager = storage_manager
        self._session_manager = session_manager

    def on_event(self, event: Dict[str, Any]) -> None:
        if event.get("type") != "session.updated":
            return

        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(self._async_write(event))
            task.add_done_callback(self._handle_write_error)
        except RuntimeError:
            logger.warning("no event loop for conversation write - event dropped")

    def _handle_write_error(self, task: asyncio.Task) -> None:
        try:
            task.result()
        except Exception as e:
            logger.error(f"conversation write failed: {e}", exc_info=True)

    async def _async_write(self, event: Dict[str, Any]) -> None:
        session_id = event.get("session_id")

        if not session_id:
            return

        conversation_id = session_id
        messages = event.get("messages") or []
        llm_context = event.get("llm_context")
        usage = event.get("usage") or {}

        delta_cost = float(usage.get("cost", 0)) if isinstance(usage, dict) else 0.0
        currency = usage.get("currency", "USD") if isinstance(usage, dict) else "USD"

        persisted_state = None
        if self._session_manager:
            try:
                session_data = await self._session_manager.get_data(session_id)
                if session_data:
                    persisted_state = session_data.get("__persisted_state__")
            except Exception as e:
                logger.warning(f"could not read persisted_state for {session_id}: {e}")

        metadata_parts: Dict[str, Any] = {}

        if event.get("conversation_metadata"):
            metadata_parts["conversation_metadata"] = event["conversation_metadata"]

        if persisted_state:
            metadata_parts["persisted_state"] = persisted_state

        metadata = json.dumps(metadata_parts) if metadata_parts else None

        try:
            turn_count = await self._storage_manager.query(
                "save_conversation",
                conversation_id=conversation_id,
                session_id=session_id,
                source=event.get("source"),
                delta_cost=delta_cost,
                currency=currency,
                organization_id=event.get("organization_id"),
                user_id=event.get("user_id"),
                application_id=event.get("application_id"),
                metadata=metadata
            )

            if messages:
                await self._storage_manager.query(
                    "save_messages",
                    messages=messages,
                    conversation_id=conversation_id,
                    turn_number=turn_count
                )

            if llm_context:
                context_messages = llm_context.get("messages") or []

                await self._storage_manager.query(
                    "save_llm_context",
                    conversation_id=conversation_id,
                    turn_number=turn_count,
                    messages=context_messages
                )

        except Exception as e:
            logger.error(f"conversation write failed for session {session_id}: {e}", exc_info=True)
