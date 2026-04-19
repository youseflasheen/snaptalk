from typing import Literal

from pydantic import BaseModel, Field


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    lang_code: str = Field(min_length=2, max_length=8, pattern=r"^[A-Za-z-]+$")
    voice: str = Field(default="default", min_length=1, max_length=64)
    speed: float = Field(default=1.0, ge=0.5, le=1.5)


class TTSResponse(BaseModel):
    audio_id: str
    engine: str
    audio_url: str
    duration_ms: int


class PronunciationRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=64)
    reference_text: str = Field(min_length=1, max_length=200)
    reference_phonemes: list[str] = Field(min_length=1, max_length=128)
    lang_code: str = Field(min_length=2, max_length=8, pattern=r"^[A-Za-z-]+$")
    audio_url: str = Field(min_length=10, max_length=2048)


class PhonemeResult(BaseModel):
    phoneme: str
    correct: bool
    score: float = Field(ge=0, le=1)
    level: Literal["red", "orange", "green"] = Field(default="red")
    start_ms: int
    end_ms: int


class AlignmentResult(BaseModel):
    insertions: list[str]
    deletions: list[str]
    substitutions: list[dict[str, str]]


class PronunciationResponse(BaseModel):
    overall_score: float = Field(ge=0, le=1)
    overall_level: Literal["red", "orange", "green"] = Field(default="red")
    per_phoneme: list[PhonemeResult]
    alignment: AlignmentResult
