from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import UUID4, BaseModel, Field

from ai_api.model.api.status import ImageDuplicateStatus


class AIGeneratedStatus(Enum):
    NOT_GENERATED = "not-generated"
    GENERATED = "generated"
    GENERATED_PROTECTED = "generated-protected"


class ProcessImageRequest(BaseModel):
    image_uuid: UUID4 = Field(..., description="Image UUID", alias="uuid")
    image: bytes = Field(..., description="Image to process")
    ai_generated_status: AIGeneratedStatus = Field(..., description="AI generated status", alias="aiGenerated")
    metadata: dict[str, Any] | None = Field(default=None, description="Metadata", alias="artworkMetadata")


class BackendPatchRequest(BaseModel):
    image_uuid: UUID4 = Field(..., description="Image UUID", serialization_alias="uuid")
    image_duplicate_status: ImageDuplicateStatus = Field(default=ImageDuplicateStatus.OK, description="Image duplicate status", serialization_alias="duplicateStatus")
    closest_match_uuid: UUID4 | None = Field(default=None, description="Closest image match UUID if duplicate or plagiarised", serialization_alias="closestMatchUuid")
    modified_image_uuid: UUID4 | None = Field(default=None, description="Modified image UUID", serialization_alias="newUuid")
    image: bytes | None = Field(default=None, description="New image", serialization_alias="image")
    ai_generated_status: AIGeneratedStatus = Field(default=None, description="AI generated status", serialization_alias="aiGenerated")
    metadata: dict[str, Any] | None = Field(default=None, description="Metadata", serialization_alias="artworkMetadata")
