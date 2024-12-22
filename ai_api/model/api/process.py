"""Module containing Pydantic models for image processing operations.

This module provides request and response models for image processing operations,
including AI generation status tracking, image processing requests, and backend
communication models.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from bentoml import Field, IODescriptor
from PIL import Image as PILImage
from pydantic import UUID4

from ai_api.model.api.status import ImageDuplicateStatus


class AIGeneratedStatus(Enum):
    """Enum representing the AI generation status of an image.

    This enum tracks whether an image was generated by AI, is a natural image,
    or is protected against AI training.

    Attributes:
        NOT_GENERATED: Image is not AI-generated (natural image).
        GENERATED: Image is AI-generated.
        GENERATED_PROTECTED: Image is AI-generated and protected against training.

    """

    NOT_GENERATED = "NOT_GENERATED"
    GENERATED = "GENERATED"
    GENERATED_PROTECTED = "GENERATED_PROTECTED"


class ProcessImageRequest(IODescriptor):
    """Request model for processing an image.

    This class represents a request to process an image, including its metadata
    and AI generation status information.

    :param image: The image to process.
    :type image: PILImage.Image
    :param image_uuid: Unique identifier for the image.
    :type image_uuid: UUID4
    :param ai_generated_status: Status indicating if/how the image was AI-generated.
    :type ai_generated_status: AIGeneratedStatus
    :param metadata: Optional metadata about the artwork.
    :type metadata: dict[str, Any] | None
    """

    image: PILImage.Image = Field(..., description="Image to process")
    image_uuid: UUID4 = Field(..., description="Image UUID", alias="uuid")
    ai_generated_status: AIGeneratedStatus = Field(..., description="AI generated status", alias="aiGenerated")
    metadata: dict[str, Any] | None = Field(default=None, description="Metadata", alias="artworkMetadata")

    class Config:
        """Pydantic configuration for the ProcessImageRequest class."""

        arbitrary_types_allowed = True


class BackendPatchRequest(IODescriptor):
    """Request model for updating image information on the backend.

    This class represents a request to update image information on the backend,
    including duplicate status, AI generation status, and related image references.

    :param image_uuid: UUID of the image to update.
    :type image_uuid: UUID4
    :param image_duplicate_status: Status indicating if image is a duplicate.
    :type image_duplicate_status: ImageDuplicateStatus
    :param closest_match_uuid: UUID of the most similar image if duplicate/plagiarized.
    :type closest_match_uuid: UUID4 | None
    :param modified_image_uuid: UUID of the modified version of the image.
    :type modified_image_uuid: UUID4 | None
    :param ai_generated_status: Status indicating if/how the image was AI-generated.
    :type ai_generated_status: AIGeneratedStatus | None
    :param metadata: Optional metadata about the artwork.
    :type metadata: dict[str, Any] | None
    """

    image_uuid: UUID4 = Field(..., description="Image UUID", serialization_alias="uuid")
    image_duplicate_status: ImageDuplicateStatus = Field(default=ImageDuplicateStatus.OK, description="Image duplicate status", serialization_alias="duplicateStatus")
    closest_match_uuid: UUID4 | None = Field(default=None, description="Closest image match UUID if duplicate or plagiarised", serialization_alias="closestMatchUuid")
    modified_image_uuid: UUID4 | None = Field(default=None, description="Modified image UUID", serialization_alias="newUuid")
    ai_generated_status: AIGeneratedStatus | None = Field(default=None, description="AI generated status", serialization_alias="aiGenerated")
    metadata: dict[str, Any] | None = Field(default=None, description="Metadata", serialization_alias="artworkMetadata")


class AddWatermarkRequest(IODescriptor):
    """Request model for adding a watermark to an image."""

    image: PILImage.Image
    watermark_text: str

    class Config:
        """Pydantic configuration for the AddWatermarkRequest class."""

        arbitrary_types_allowed = True


class ListAddWatermarkRequest(IODescriptor):
    """Request model for adding a watermark to a list of images."""

    images: list[AddWatermarkRequest]

    class Config:
        """Pydantic configuration for the ListAddWatermarkRequest class."""

        arbitrary_types_allowed = True
