from pydantic import UUID4, BaseModel, Field


class ImageSearchRequest(BaseModel):
    image_uuid: UUID4 = Field(..., description="UUID of image to find similar images to")
    count: int = Field(default=50, description="Number of results to return") 
    page: int = Field(default=0, description="Page number to return")


class ImageSearchResponse(BaseModel):
    image_uuid: list[UUID4] = Field(..., description="UUIDs of the most similar images")


class RawImageSearchRequest(BaseModel):
    image: bytes = Field(..., description="Raw image bytes")
    count: int = Field(default=50, description="Number of results to return")
    page: int = Field(default=0, description="Page number to return")
