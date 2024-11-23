from enum import Enum

from pydantic import BaseModel, Field

from ai_api.model.api.status import ImageDuplicateStatus


class AIGeneratedStatus(Enum):
    NOT_GENERATED = "not-generated"
    GENERATED = "generated"
    GENERATED_PROTECTED = "generated-protected"


class ProcessImageRequest(BaseModel):
    image_id: int = Field(..., description="Image ID", alias="id")
    image: bytes = Field(..., description="Image to process")
    ai_generated_status: AIGeneratedStatus = Field(..., description="AI generated status", alias="aiGenerated")
    metadata: dict = Field(..., description="Metadata", alias="artworkMetadata")


class BackendPatchRequest(BaseModel):
    image_id: int = Field(..., description="Image ID", serialization_alias="id")
    image_duplicate_status: ImageDuplicateStatus = Field(default=ImageDuplicateStatus.OK, description="Image duplicate status", serialization_alias="duplicateStatus")
    closest_match_id: int | None = Field(default=None, description="Closest image match ID if duplicate or plagiarised", serialization_alias="closestMatchId")
    modified_image_id: int = Field(default=None, description="Modified image ID", serialization_alias="newId")
    image: bytes = Field(default=None, description="New image", serialization_alias="image")
    ai_generated_status: AIGeneratedStatus = Field(default=None, description="AI generated status", serialization_alias="aiGenerated")
    metadata: dict = Field(default=None, description="Metadata", serialization_alias="artworkMetadata")
