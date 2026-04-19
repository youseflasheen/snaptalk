import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core.config import settings
from app.schemas.pipeline import SnapLearnResponse
from app.services.snap_learn_service import run_snap_learn

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/snap-learn", response_model=SnapLearnResponse)
async def snap_learn(
    image: UploadFile = File(...),
    target_lang: str = Form("es"),
    max_objects: int = Form(3),
) -> SnapLearnResponse:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid content type")

    if max_objects < 1 or max_objects > 10:
        raise HTTPException(status_code=422, detail="max_objects must be in [1,10]")

    max_bytes = settings.max_upload_image_bytes
    data = await image.read(max_bytes + 1)
    if not data:
        raise HTTPException(status_code=422, detail="empty image")
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail="image exceeds maximum allowed size")

    try:
        return run_snap_learn(image_bytes=data, target_lang=target_lang, max_objects=max_objects)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Invalid image payload") from exc
    except Exception as exc:
        logger.exception("snap_learn endpoint failed")
        raise HTTPException(status_code=500, detail="Internal error") from exc
