"""empty message

Revision ID: b4328055fa2f
Revises: 73cbf5c3801e
Create Date: 2024-08-16 19:28:39.010021

"""
from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from ai_api import settings
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b4328055fa2f"
down_revision: Union[str, None] = "73cbf5c3801e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create image table with merged columns
    op.create_table("image",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("image_uuid", sa.UUID(as_uuid=True), nullable=False),
        sa.Column("original_image_uuid", sa.UUID(as_uuid=True), nullable=False),
        sa.Column("generated_status", sa.Enum("AI_GENERATED", "NOT_AI_GENERATED", name="aigeneratedstatus"), nullable=False),
        sa.Column("image_metadata", sa.JSON(), nullable=False),
        sa.Column("image_hash", sa.String(length=8), nullable=False),
        sa.Column("user_annotation", sa.String(), nullable=True),
        sa.Column("generated_annotation", sa.String(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("image_uuid"),
        sa.UniqueConstraint("original_image_uuid"),
    )

    # Add indexes
    op.create_index("idx_image_uuid", "image", ["image_uuid"])
    op.create_index("idx_image_hash", "image", ["image_hash"])

    # Create gallery_embedding table
    op.create_table("gallery_embedding",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("image_id", sa.Integer(), nullable=False),
        sa.Column("image_embedding", Vector(settings.model.embedding.dimension), nullable=False),
        sa.Column("watermarked_image_embedding", Vector(settings.model.embedding.dimension), nullable=False),
        sa.Column("metadata_embedding", Vector(settings.model.embedding.dimension), nullable=False),
        sa.Column("user_caption_embedding", Vector(settings.model.embedding.dimension), nullable=True),
        sa.Column("generated_caption_embedding", Vector(settings.model.embedding.dimension), nullable=True),
        sa.ForeignKeyConstraint(["image_id"], ["image.id"]),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("gallery_embedding")
    op.drop_table("image")
