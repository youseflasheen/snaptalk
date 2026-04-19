from typing import Literal

from pydantic import ConfigDict
from pydantic import BaseModel, Field


class VisionDetectRequest(BaseModel):
    image_url: str = Field(min_length=10, max_length=2048)
    max_objects: int = Field(default=5, ge=1, le=10)
    language: str = Field(default="en", min_length=2, max_length=8, pattern=r"^[A-Za-z-]+$")


class VisionObject(BaseModel):
    object_id: str
    bbox: list[int] = Field(description="[x_min, y_min, x_max, y_max] in pixels")
    polygon: list[list[int]] = Field(description="List of [x, y] points in pixels")
    detector_label: str
    canonical_tag: str
    confidence: float = Field(ge=0, le=1)
    source: Literal["rampp", "fallback"] = "rampp"


class VisionDetectResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    request_id: str
    model_version: str
    image_width: int
    image_height: int
    objects: list[VisionObject]
    latency_ms: int
    fallback_used: bool = False
