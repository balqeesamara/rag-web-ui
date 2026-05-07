from fastapi import APIRouter
from app.api.api_v1 import auth, knowledge_base, chat, query
from app.core.config import settings

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(knowledge_base.router, prefix="/knowledge-base", tags=["knowledge-base"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(query.router, prefix="/query", tags=["query"])


@api_router.get("/config", tags=["config"])
def get_client_config():
    """Expose non-sensitive runtime configuration to the frontend."""
    return {
        "chunk_size": settings.CHUNK_SIZE,
        "chunk_overlap": settings.chunk_overlap,
    }