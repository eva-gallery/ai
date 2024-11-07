from typing import Optional

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., description="Query to search for")
    count: Optional[int] = Field(default=50, description="Number of results to return")
    page: Optional[int] = Field(default=0, description="Page number to return")


class SearchResponse(BaseModel):
    id_list: list[int] = Field(..., description="List of image IDs")
