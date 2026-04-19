from fastapi import APIRouter

from app.schemas.speech import PronunciationRequest, PronunciationResponse
from app.services.pronunciation_service import score_pronunciation

router = APIRouter()


@router.post("/pronunciation", response_model=PronunciationResponse)
def pronunciation(req: PronunciationRequest) -> PronunciationResponse:
    return score_pronunciation(req)
