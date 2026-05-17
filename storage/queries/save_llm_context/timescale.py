import hashlib
import json
from typing import Any, Dict, List

from optorch.storage.queries.base import BaseQuery
from optorch.logging import get_logger

logger = get_logger(__name__)


class SaveLlmContextQuery(BaseQuery):

    @property
    def query_name(self) -> str:
        return "save_llm_context"

    async def execute(
        self,
        conversation_id: str,
        turn_number: int,
        messages: List[Dict[str, Any]],
    ) -> None:
        insert_message = """
            INSERT INTO messages (
                id, conversation_id, layer, turn_number, sequence_order, role, content
            ) VALUES (
                :id, :conversation_id, 'llm_context', :turn_number, :sequence_order, :role, :content
            )
            ON CONFLICT (id) DO NOTHING
        """

        insert_ref = """
            INSERT INTO llm_context_refs (
                conversation_id, turn_number, message_id, sequence_order
            ) VALUES (
                :conversation_id, :turn_number, :message_id, :sequence_order
            )
            ON CONFLICT (conversation_id, turn_number, message_id) DO NOTHING
        """

        for seq, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if not isinstance(content, str):
                content = json.dumps(content)

            msg_id = hashlib.sha256(
                f"{conversation_id}|{role}|{content}".encode()
            ).hexdigest()[:36]

            await self.store.execute(insert_message, {
                "id": msg_id,
                "conversation_id": conversation_id,
                "turn_number": turn_number,
                "sequence_order": seq,
                "role": role,
                "content": content,
            })

            await self.store.execute(insert_ref, {
                "conversation_id": conversation_id,
                "turn_number": turn_number,
                "message_id": msg_id,
                "sequence_order": seq,
            })

        logger.debug(f"saved llm context for conversation {conversation_id}, turn {turn_number}")
