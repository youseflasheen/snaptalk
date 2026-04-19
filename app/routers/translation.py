from fastapi import APIRouter

from app.schemas.translation import FlashcardRequest, FlashcardResponse
from app.services.translation_service import build_flashcard

router = APIRouter()


@router.post("/flashcard", response_model=FlashcardResponse)
def flashcard(req: FlashcardRequest) -> FlashcardResponse:
    return build_flashcard(req)
