"""The module with Pydantic models for the status API."""

from __future__ import annotations

from enum import Enum

from bentoml import IODescriptor


class ImageDuplicateStatus(Enum):
    """Whether the image is a duplicate or plagiarised."""

    OK = "ok"
    EXISTS = "exists"
    PLAGIARIZED = "plagiarized"

class ImageDuplicateResponse(IODescriptor):
    """The response to check if an image is a duplicate or plagiarised."""

    status: ImageDuplicateStatus
