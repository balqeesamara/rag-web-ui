from pydantic import BaseModel, model_validator, field_serializer
from typing import Any, List, Optional
from datetime import datetime


def _as_utc_iso(dt: datetime) -> str:
    if dt is None:
        return None
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


class MessageBase(BaseModel):
    content: str
    role: str

class MessageCreate(MessageBase):
    chat_id: int

class MessageResponse(MessageBase):
    id: int
    chat_id: int
    created_at: datetime
    updated_at: datetime

    @field_serializer("created_at", "updated_at")
    def serialise_datetimes(self, v): return _as_utc_iso(v)

    class Config:
        from_attributes = True

class ChatBase(BaseModel):
    title: str

class ChatCreate(ChatBase):
    knowledge_base_ids: List[int]
    use_graph_rag: bool = False

class ChatUpdate(ChatBase):
    knowledge_base_ids: Optional[List[int]] = None

class ChatResponse(ChatBase):
    id: int
    user_id: int
    use_graph_rag: bool = False
    created_at: datetime
    updated_at: datetime
    messages: List[MessageResponse] = []
    knowledge_base_ids: List[int] = []

    @field_serializer("created_at", "updated_at")
    def serialise_datetimes(self, v): return _as_utc_iso(v)

    @model_validator(mode='before')
    @classmethod
    def extract_kb_ids(cls, data: Any) -> Any:
        if hasattr(data, 'knowledge_bases'):
            data.__dict__['knowledge_base_ids'] = [kb.id for kb in data.knowledge_bases]
        return data

    class Config:
        from_attributes = True
