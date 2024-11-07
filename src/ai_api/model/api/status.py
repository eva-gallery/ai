from enum import Enum

from pydantic import BaseModel


class ResponseType(Enum):
    OK = "ok"
    EXISTS = "exists"

class ResponseStatus(BaseModel):
    status: ResponseType
