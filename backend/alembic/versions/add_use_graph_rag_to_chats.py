"""add use_graph_rag to chats

Revision ID: add_use_graph_rag_to_chats
Revises: add_chat_history_summary
Create Date: 2025-01-01 00:00:01.000000
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = 'add_use_graph_rag_to_chats'
down_revision: Union[str, None] = 'add_chat_history_summary'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "chats",
        sa.Column(
            "use_graph_rag",
            sa.Boolean(),
            nullable=False,
            server_default="0",
        ),
    )


def downgrade() -> None:
    op.drop_column("chats", "use_graph_rag")
