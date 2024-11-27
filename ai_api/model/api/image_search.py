from pydantic import BaseModel, Field


class ImageSearchRequest(BaseModel):
    image_id: int = Field(..., description="ID of image to find similar images to")
    count: int = Field(default=50, description="Number of results to return") 
    page: int = Field(default=0, description="Page number to return")


class ImageSearchResponse(BaseModel):
    image_id: list[int] = Field(..., description="Image IDs of the most similar images")


class RawImageSearchRequest(BaseModel):
    image: bytes = Field(..., description="Raw image bytes")
    count: int = Field(default=50, description="Number of results to return")
    page: int = Field(default=0, description="Page number to return")
