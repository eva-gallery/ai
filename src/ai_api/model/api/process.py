from enum import Enum

from pydantic import BaseModel, Field


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
    image_uuid: str = Field(..., description="Image UUID", serialization_alias="id")
    modified_image_uuid: str = Field(..., description="Modified image UUID", serialization_alias="newId")
    image: bytes = Field(..., description="New image")
    ai_generated_status: AIGeneratedStatus = Field(..., description="AI generated status", serialization_alias="aiGenerated")
    metadata: dict = Field(..., description="Metadata", serialization_alias="artworkMetadata")
