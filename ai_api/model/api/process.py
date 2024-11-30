"""The module with Pydantic models for the process API."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

from bentoml import Field, IODescriptor

from ai_api.model.api.status import ImageDuplicateStatus

if TYPE_CHECKING:
    from PIL import Image as PILImage
    from pydantic import UUID4


class AIGeneratedStatus(Enum):
    """Whether the image is AI generated, or protected by AI against training."""

    NOT_GENERATED = "not-generated"
    GENERATED = "generated"
    GENERATED_PROTECTED = "generated-protected"


class ProcessImageRequest(IODescriptor):
    """The request to process an image."""

    image: PILImage.Image = Field(..., description="Image to process")
    image_uuid: UUID4 = Field(..., description="Image UUID", alias="uuid")
    ai_generated_status: AIGeneratedStatus = Field(..., description="AI generated status", alias="aiGenerated")
    metadata: dict[str, Any] | None = Field(default=None, description="Metadata", alias="artworkMetadata")


class BackendPatchRequest(IODescriptor):
    """The request to patch an image on the backend."""

    image_uuid: UUID4 = Field(..., description="Image UUID", serialization_alias="uuid")
    image_duplicate_status: ImageDuplicateStatus = Field(default=ImageDuplicateStatus.OK, description="Image duplicate status", serialization_alias="duplicateStatus")
    closest_match_uuid: UUID4 | None = Field(default=None, description="Closest image match UUID if duplicate or plagiarised", serialization_alias="closestMatchUuid")
    modified_image_uuid: UUID4 | None = Field(default=None, description="Modified image UUID", serialization_alias="newUuid")
    ai_generated_status: AIGeneratedStatus = Field(default=None, description="AI generated status", serialization_alias="aiGenerated")
    metadata: dict[str, Any] | None = Field(default=None, description="Metadata", serialization_alias="artworkMetadata")
