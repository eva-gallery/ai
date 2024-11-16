from enum import Enum

from pydantic import BaseModel


class ImageDuplicateStatus(Enum):
    OK = "ok"
    EXISTS = "exists"
    PLAGIARIZED = "plagiarized"

class ImageDuplicateResponse(BaseModel):
    status: ImageDuplicateStatus
