from typing import Literal

from pydantic import BaseModel, Field


class FlashcardRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=64)
    object_id: str = Field(min_length=1, max_length=64)
    source_word: str = Field(min_length=1, max_length=80)
    source_lang: str = Field(default="en", min_length=2, max_length=8, pattern=r"^[A-Za-z-]+$")
    target_lang: str = Field(min_length=2, max_length=8, pattern=r"^[A-Za-z-]+$")
    proficiency_level: str = Field(default="A2", min_length=1, max_length=16)


class FlashcardResponse(BaseModel):
    flashcard_id: str
    source_word: str
    translated_word: str
    ipa: str
    example_sentence: str
    example_translation: str
    translation_source: Literal[
        "translation_memory",  # From SQLite cache
        "google_cloud",        # Official Google Cloud Translation API
        "google",              # Google Translate (via deep-translator)
        "deepl",               # DeepL API
        "mymemory",            # MyMemory translation fallback
        "ollama",              # Ollama LLM fallback
        "human_verified",      # User-verified translation
        "error_fallback"       # Fallback on error
    ]
    cached: bool
