"""Added public column to image table, with default value of false.

Revision ID: f86ad1c80fd3
Revises: b4328055fa2f
Create Date: 2024-12-22 12:00:00.000000

"""
from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f86ad1c80fd3"
down_revision: Union[str, None] = "b4328055fa2f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add public column to image table with default value of false."""
    op.add_column("image", sa.Column("public", sa.Boolean(), server_default="false", nullable=False))


def downgrade() -> None:
    """Remove public column from image table."""
    op.drop_column("image", "public")
