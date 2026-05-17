from typing import Any, Dict, Optional
from optorch.storage.queries.base import BaseQuery
from optorch.logging import get_logger

logger = get_logger(__name__)

class GetConversationQuery(BaseQuery):

    @property
    def query_name(self) -> str:
        return "get_conversation"

    async def execute(self, session_id: str) -> Optional[Dict[str, Any]]:
        query = """
            SELECT
                id, session_id, turn_count, total_cost, currency,
                organization_id, user_id, application_id, metadata, created_at, updated_at
            FROM conversations
            WHERE session_id = :session_id
        """
        row = await self.store.fetch_one(query, {"session_id": session_id})
        return dict(row) if row else None
