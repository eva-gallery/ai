from bentoml import Field, IODescriptor
from PIL.Image import Image as PILImage


class EmbedRequest(IODescriptor):
    """The request to embed images."""

    image: list[PILImage] = Field(..., description="List of images to embed")

    class Config:
        arbitrary_types_allowed = True
