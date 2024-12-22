"""Module containing status-related enums and response models for image duplicate checking.

This module provides enums and response models to handle image duplicate status checks,
including plagiarism detection and duplicate image identification.
"""

from __future__ import annotations

from enum import Enum

from bentoml import IODescriptor


class ImageDuplicateStatus(Enum):
    """Enum representing the duplicate or plagiarism status of an image.

    This enum is used to indicate whether an image is unique, already exists in the system,
    or has been identified as plagiarized content.

    Attributes:
        OK: Image is unique and acceptable.
        EXISTS: Image is a duplicate of an existing image.
        PLAGIARIZED: Image is identified as plagiarized content.

    """

    OK = "OK"
    EXISTS = "EXISTS"
    PLAGIARIZED = "PLAGIARIZED"


class ImageDuplicateResponse(IODescriptor):
    """Response model for image duplicate checking.

    This class represents the response structure for image duplicate checking operations,
    containing the status of the duplicate check.

    :param status: The duplicate status of the checked image.
    :type status: ImageDuplicateStatus
    """

    status: ImageDuplicateStatus
