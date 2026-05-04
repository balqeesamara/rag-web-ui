"""add history_summary to chats

Revision ID: add_chat_history_summary
Revises: a1b2c3d4e5f6
Create Date: 2025-01-01 00:00:00.000000
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

revision: str = 'add_chat_history_summary'
down_revision: Union[str, None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'chats',
        sa.Column('history_summary', mysql.LONGTEXT(), nullable=True)
    )


def downgrade() -> None:
    op.drop_column('chats', 'history_summary')
