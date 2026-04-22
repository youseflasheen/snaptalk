from fastapi import APIRouter

from app.schemas.speech import TTSRequest, TTSResponse
from app.services.tts.service import synthesize

router = APIRouter()


@router.post("/tts", response_model=TTSResponse)
def tts(req: TTSRequest) -> TTSResponse:
    return synthesize(req)
