"""empty message

Revision ID: b4328055fa2f
Revises: 73cbf5c3801e
Create Date: 2024-08-16 19:28:39.010021

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = 'b4328055fa2f'
down_revision: Union[str, None] = '73cbf5c3801e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create image_data table
    op.create_table('image_data',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('original_image_id', sa.Integer(), nullable=False),
        sa.Column('image_metadata', sa.JSON(), nullable=False),
        sa.Column('md5_hash', sa.String(length=32), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )

    # Create annotation table
    op.create_table('annotation',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_annotation', sa.String(), nullable=True),
        sa.Column('generated_annotation', sa.String(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )

    # Create image table
    op.create_table('image',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('image_data_id', sa.Integer(), nullable=False),
        sa.Column('annotation_id', sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(['annotation_id'], ['annotation.id'], ),
        sa.ForeignKeyConstraint(['image_data_id'], ['image_data.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # Create gallery_embedding table
    op.create_table('gallery_embedding',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('image_id', sa.Integer(), nullable=False),
        sa.Column('image_embedding', Vector(1536), nullable=False),
        sa.Column('watermarked_image_embedding', Vector(1536), nullable=False),
        sa.Column('metadata_embedding', Vector(512), nullable=False),
        sa.Column('user_caption_embedding', Vector(512), nullable=True),
        sa.Column('generated_caption_embedding', Vector(512), nullable=True),
        sa.ForeignKeyConstraint(['image_id'], ['image.id'], ),
        sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    op.drop_table('gallery_embedding')
    op.drop_table('image')
    op.drop_table('annotation')
    op.drop_table('image_data')
