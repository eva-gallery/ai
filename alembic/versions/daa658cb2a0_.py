"""Drop image_hash column and update enum.

Revision ID: daa658cb2a0
Revises: d8c1f6a0457e
Create Date: 2025-01-28 12:00:00.000000

"""
from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "daa658cb2a0"
down_revision: Union[str, None] = "d8c1f6a0457e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Update enum type to match new values
    op.execute("ALTER TYPE aigeneratedstatus RENAME TO aigeneratedstatus_old")
    op.execute("CREATE TYPE aigeneratedstatus AS ENUM ('NOT_GENERATED', 'GENERATED', 'GENERATED_PROTECTED')")
    op.execute("""
        ALTER TABLE image 
        ALTER COLUMN generated_status TYPE aigeneratedstatus 
        USING CASE 
            WHEN generated_status::text = 'NOT_AI_GENERATED' THEN 'NOT_GENERATED'::aigeneratedstatus
            WHEN generated_status::text = 'AI_GENERATED' THEN 'GENERATED'::aigeneratedstatus
        END
    """)
    op.execute("DROP TYPE aigeneratedstatus_old")

    # Drop image_hash column and its index
    op.drop_index("idx_image_hash")
    op.drop_column("image", "image_hash")


def downgrade() -> None:
    # Restore image_hash column and index
    op.add_column("image", sa.Column("image_hash", sa.BYTEA(), nullable=True))
    op.create_index("idx_image_hash", "image", ["image_hash"])

    # Restore old enum values
    op.execute("ALTER TYPE aigeneratedstatus RENAME TO aigeneratedstatus_old")
    op.execute("CREATE TYPE aigeneratedstatus AS ENUM ('NOT_AI_GENERATED', 'AI_GENERATED')")
    op.execute("""
        ALTER TABLE image 
        ALTER COLUMN generated_status TYPE aigeneratedstatus 
        USING CASE 
            WHEN generated_status::text = 'NOT_GENERATED' THEN 'NOT_AI_GENERATED'::aigeneratedstatus
            WHEN generated_status::text = 'GENERATED' THEN 'AI_GENERATED'::aigeneratedstatus
            WHEN generated_status::text = 'GENERATED_PROTECTED' THEN 'AI_GENERATED'::aigeneratedstatus
        END
    """)
    op.execute("DROP TYPE aigeneratedstatus_old")
