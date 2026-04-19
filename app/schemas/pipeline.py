from pydantic import BaseModel, Field


class SnapLearnObject(BaseModel):
    object_id: str
    bbox: list[int] = Field(description="[x_min, y_min, x_max, y_max] in pixels")
    polygon: list[list[int]] = Field(description="List of [x, y] contour points")
    canonical_tag: str
    confidence: float = Field(ge=0, le=1)
    translated_word: str
    ipa: str
    masked_image_base64: str = Field(description="Base64-encoded PNG of isolated object with transparent background")


class SnapLearnResponse(BaseModel):
    image_width: int
    image_height: int
    objects: list[SnapLearnObject]
    total_objects: int
