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
        sa.Column('original_image_uuid', sa.UUID(as_uuid=True), nullable=False),
        sa.Column('modified_image_uuid', sa.UUID(as_uuid=True), nullable=True),
        sa.Column('image_metadata', sa.JSON(), nullable=True),
        sa.Column('md5_hash', sa.String(length=32), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )

    # Create annotation table
    op.create_table('annotation',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_annotation', sa.String(), nullable=True),
        sa.Column('generated_annotation', sa.String(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )

    # Create captioning_model table
    op.create_table('captioning_model',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )

    # Create image_embedding_model table
    op.create_table('image_embedding_model',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('vector_length', Vector(1536), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )

    # Create text_embedding_model table
    op.create_table('text_embedding_model',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('vector_length', Vector(512), nullable=False),
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
        sa.Column('image_embedding_model_id', sa.Integer(), nullable=False),
        sa.Column('text_embedding_model_id', sa.Integer(), nullable=False),
        sa.Column('captioning_model_id', sa.Integer(), nullable=False),
        sa.Column('image_embedding', Vector(1536), nullable=False),
        sa.Column('watermarked_image_embedding', Vector(1536), nullable=False),
        sa.Column('text_embedding', Vector(512), nullable=False),
        sa.Column('metadata_embedding', Vector(512), nullable=False),
        sa.Column('image_id', sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(['captioning_model_id'], ['captioning_model.id'], ),
        sa.ForeignKeyConstraint(['image_embedding_model_id'], ['image_embedding_model.id'], ),
        sa.ForeignKeyConstraint(['image_id'], ['image.id'], ),
        sa.ForeignKeyConstraint(['text_embedding_model_id'], ['text_embedding_model.id'], ),
        sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    op.drop_table('gallery_embedding')
    op.drop_table('image')
    op.drop_table('text_embedding_model')
    op.drop_table('image_embedding_model')
    op.drop_table('captioning_model')
    op.drop_table('annotation')
    op.drop_table('image_data')
