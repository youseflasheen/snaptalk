"""VLM-based Snap & Learn endpoint using Qwen2-VL for labeling."""
import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core.config import settings
from app.schemas.pipeline import SnapLearnResponse
from app.services.vlm_experiment.snap_learn_vlm import run_snap_learn_vlm

router = APIRouter(prefix="/v1/pipeline", tags=["pipeline-vlm"])
logger = logging.getLogger(__name__)


@router.post("/snap-learn-vlm", response_model=SnapLearnResponse)
async def snap_learn_vlm_endpoint(
    image: UploadFile = File(...),
    target_lang: str = Form(...),
    max_objects: int = Form(5),
) -> SnapLearnResponse:
    """
    VLM-based Snap & Learn pipeline using Qwen2-VL for accurate labeling.

    Environment variables: DETECTOR (yolo/fastsam/ldet), SEGMENTATION (bbox/fastsam/mobilesam)
    """
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid content type")

    if max_objects < 1 or max_objects > 10:
        raise HTTPException(status_code=400, detail="max_objects must be 1-10")

    max_bytes = settings.max_upload_image_bytes
    image_bytes = await image.read(max_bytes + 1)
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty image file")
    if len(image_bytes) > max_bytes:
        raise HTTPException(status_code=413, detail="image exceeds maximum allowed size")

    try:
        return run_snap_learn_vlm(
            image_bytes=image_bytes,
            target_lang=target_lang,
            max_objects=max_objects,
        )
    except RuntimeError:
        raise HTTPException(status_code=503, detail="VLM provider unavailable")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid image payload")
    except Exception as exc:
        logger.exception("snap_learn_vlm endpoint failed")
        raise HTTPException(status_code=500, detail="Internal error") from exc
