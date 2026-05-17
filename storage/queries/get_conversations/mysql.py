from typing import Any, Dict, List, Optional
from optorch.storage.queries.base import BaseQuery
from optorch.logging import get_logger

logger = get_logger(__name__)

class GetConversationsQuery(BaseQuery):

    @property
    def query_name(self) -> str:
        return "get_conversations"

    async def execute(
        self,
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        query = """
            SELECT
                id, session_id, turn_count, total_cost, currency,
                organization_id, user_id, application_id, created_at, updated_at
            FROM conversations
            WHERE (:organization_id IS NULL OR organization_id = :organization_id)
              AND (:user_id IS NULL OR user_id = :user_id)
            ORDER BY created_at DESC
            LIMIT :limit OFFSET :offset
        """

        rows = await self.store.fetch_all(query, {
            "organization_id": organization_id,
            "user_id": user_id,
            "limit": limit,
            "offset": offset,
        })

        return [dict(row) for row in rows]
