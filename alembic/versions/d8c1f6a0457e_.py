"""Change image_hash column type from string to bytea with backup.

Revision ID: d8c1f6a0457e
Revises: f86ad1c80fd3
Create Date: 2025-01-07 12:00:00.000000

"""
from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d8c1f6a0457e"
down_revision: Union[str, None] = "f86ad1c80fd3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create backup table
    op.create_table(
        "image_hashes_backup",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("image_hash", sa.String(8)),
        sa.ForeignKeyConstraint(["id"], ["image.id"], ondelete="CASCADE"),
    )

    # Copy data to backup
    op.execute(
        """
        INSERT INTO image_hashes_backup (id, image_hash)
        SELECT id, image_hash FROM image WHERE image_hash IS NOT NULL
        """
    )

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

    # Restore data from backup
    op.execute(
        """
        UPDATE image i
        SET image_hash = b.image_hash
        FROM image_hashes_backup b
        WHERE i.id = b.id
        """
    )

    # Recreate original index
    op.create_index("idx_image_hash", "image", ["image_hash"])

    # Drop backup table
    op.drop_table("image_hashes_backup")
