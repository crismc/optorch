import json
from typing import Any, Dict, List
from optorch.storage.queries.base import BaseQuery
from optorch.logging import get_logger

logger = get_logger(__name__)


class SaveMessagesQuery(BaseQuery):

    @property
    def query_name(self) -> str:
        return "save_messages"

    async def execute(
        self,
        messages: List[Dict[str, Any]],
        conversation_id: str,
        turn_number: int,
    ) -> None:
        query = """
            INSERT OR IGNORE INTO messages (
                id, conversation_id, layer, turn_number, sequence_order,
                role, content, model, input_tokens, output_tokens, cost,
                node_name, capabilities, metadata
            ) VALUES (
                :id, :conversation_id, :layer, :turn_number, :sequence_order,
                :role, :content, :model, :input_tokens, :output_tokens, :cost,
                :node_name, :capabilities, :metadata
            )
        """

        for i, msg in enumerate(messages):
            capabilities = msg.get("capabilities")
            metadata = msg.get("metadata")

            values = {
                "id": msg["id"],
                "conversation_id": conversation_id,
                "layer": msg.get("layer", "thread"),
                "turn_number": turn_number,
                "sequence_order": i,
                "role": msg["role"],
                "content": msg["content"],
                "model": msg.get("model"),
                "input_tokens": msg.get("input_tokens"),
                "output_tokens": msg.get("output_tokens"),
                "cost": msg.get("cost"),
                "node_name": msg.get("node_name"),
                "capabilities": json.dumps(capabilities) if capabilities else None,
                "metadata": json.dumps(metadata) if metadata else None,
            }

            await self.store.execute(query, values)

        logger.debug(f"saved {len(messages)} messages to conversation {conversation_id}, turn {turn_number}")
