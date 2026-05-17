from optorch.storage.queries.base import BaseQuery
from optorch.logging import get_logger

logger = get_logger(__name__)


class DeleteConversationQuery(BaseQuery):

    @property
    def query_name(self) -> str:
        return "delete_conversation"

    async def execute(self, session_id: str) -> None:
        await self.store.execute(
            "DELETE FROM llm_context_refs WHERE conversation_id = :session_id",
            {"session_id": session_id},
        )

        await self.store.execute(
            "DELETE FROM messages WHERE conversation_id = :session_id",
            {"session_id": session_id},
        )

        await self.store.execute(
            "DELETE FROM conversations WHERE id = :session_id",
            {"session_id": session_id},
        )
        
        logger.debug(f"deleted conversation {session_id}")
