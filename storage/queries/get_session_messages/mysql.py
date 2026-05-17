from typing import Any, Dict, List
from optorch.storage.queries.base import BaseQuery
from optorch.logging import get_logger

logger = get_logger(__name__)

class GetSessionMessagesQuery(BaseQuery):

    @property
    def query_name(self) -> str:
        return "get_session_messages"

    async def execute(
        self,
        session_id: str,
        layer: str = "thread",
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        query = """
            SELECT
                id, layer, turn_number, sequence_order, role, content,
                model, input_tokens, output_tokens, cost, node_name,
                capabilities, metadata, created_at
            FROM messages
            WHERE conversation_id = :session_id
              AND layer = :layer
            ORDER BY turn_number ASC, sequence_order ASC
            LIMIT :limit
        """

        rows = await self.store.fetch_all(query, {
            "session_id": session_id,
            "layer": layer,
            "limit": limit,
        })

        return [dict(row) for row in rows]
