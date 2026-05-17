from typing import Optional
from optorch.storage.queries.base import BaseQuery
from optorch.logging import get_logger

logger = get_logger(__name__)

class SaveConversationQuery(BaseQuery):

    @property
    def query_name(self) -> str:
        return "save_conversation"

    async def execute(
        self,
        conversation_id: str,
        session_id: str,
        source: Optional[str] = None,
        delta_cost: float = 0.0,
        currency: str = "USD",
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None,
        application_id: Optional[str] = None,
        metadata: Optional[str] = None,
    ) -> int:
        insert_query = """
            INSERT INTO conversations (
                id, session_id, source, turn_count, total_cost, currency,
                organization_id, user_id, application_id, metadata
            ) VALUES (
                :id, :session_id, :source, 1, :delta_cost, :currency,
                :organization_id, :user_id, :application_id, :metadata
            )
            ON DUPLICATE KEY UPDATE
                turn_count = turn_count + 1,
                total_cost = total_cost + :delta_cost,
                updated_at = NOW(),
                metadata = COALESCE(VALUES(metadata), metadata)
        """

        values = {
            "id": conversation_id,
            "session_id": session_id,
            "source": source,
            "delta_cost": delta_cost,
            "currency": currency,
            "organization_id": organization_id,
            "user_id": user_id,
            "application_id": application_id,
            "metadata": metadata,
        }

        await self.store.execute(insert_query, values)

        select_query = "SELECT turn_count FROM conversations WHERE session_id = :session_id"
        turn_count = await self.store.fetch_val(select_query, {"session_id": session_id})

        logger.debug(f"upserted conversation {session_id}, turn {turn_count}")

        return turn_count or 1
