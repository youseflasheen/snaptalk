from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.schemas.vision import VisionDetectRequest, VisionDetectResponse
from app.services.vision_pipeline import detect_objects
from app.utils.network_security import parse_allowed_hosts, validate_external_url

router = APIRouter()


@router.post("/detect", response_model=VisionDetectResponse)
def detect(req: VisionDetectRequest) -> VisionDetectResponse:
    try:
        validate_external_url(
            req.image_url,
            allowed_hosts=parse_allowed_hosts(settings.allowed_external_hosts),
            allow_private=settings.allow_private_network_urls,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        return detect_objects(req)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
