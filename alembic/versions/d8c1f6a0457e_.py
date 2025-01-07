"""Change image_hash column type from string to bytea with backup.

Revision ID: d8c1f6a0457e
Revises: f86ad1c80fd3
Create Date: 2025-01-07 12:00:00.000000

"""
from collections.abc import Sequence
from typing import Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d8c1f6a0457e"
down_revision: Union[str, None] = "f86ad1c80fd3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop existing indexes that depend on the column
    op.drop_index("idx_image_hash")

    # Alter column type and set all to null
    op.execute("ALTER TABLE image ALTER COLUMN image_hash TYPE bytea USING NULL")

    # Recreate the index
    op.create_index("idx_image_hash", "image", ["image_hash"])


def downgrade() -> None:
    # Drop new index
    op.drop_index("idx_image_hash")

    # Change column back to string
    op.execute("ALTER TABLE image ALTER COLUMN image_hash TYPE character varying(8) USING NULL")

    # Recreate original index
    op.create_index("idx_image_hash", "image", ["image_hash"])
