"""add retrieval leg flags to chats

Revision ID: b3c4d5e6f7a8
Revises: add_use_graph_rag_to_chats
Create Date: 2026-05-07

Adds three boolean columns to `chats`:
  use_dense    — dense vector (Qdrant cosine)   default True
  use_sparse   — sparse vector (SPLADE)         default True
  use_exact    — keyword / MySQL FTS            default True

use_graph_rag already exists; it stays as-is.
"""
from alembic import op
import sqlalchemy as sa

revision = "b3c4d5e6f7a8"
down_revision = "add_use_graph_rag_to_chats"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("chats", sa.Column("use_dense",  sa.Boolean(), nullable=False, server_default="1"))
    op.add_column("chats", sa.Column("use_sparse", sa.Boolean(), nullable=False, server_default="1"))
    op.add_column("chats", sa.Column("use_exact",  sa.Boolean(), nullable=False, server_default="1"))


def downgrade() -> None:
    op.drop_column("chats", "use_exact")
    op.drop_column("chats", "use_sparse")
    op.drop_column("chats", "use_dense")
