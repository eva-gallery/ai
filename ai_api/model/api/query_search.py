from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., description="Text query to search for")
    count: int = Field(default=50, description="Number of results to return")
    page: int = Field(default=0, description="Page number to return")


class SearchResponse(BaseModel):
    image_id: list[int] = Field(..., description="Image IDs of the most similar images")
