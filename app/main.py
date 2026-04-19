import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.routers.pipeline import router as pipeline_router
from app.routers.pipeline_vlm import router as pipeline_vlm_router
from app.routers.pronunciation import router as pronunciation_router
from app.routers.translation import router as translation_router
from app.routers.tts import router as tts_router
from app.routers.vision import router as vision_router

app = FastAPI(title=settings.app_name, version="1.0.0")

os.makedirs(settings.tts_output_dir, exist_ok=True)
app.mount("/static/audio", StaticFiles(directory=settings.tts_output_dir), name="static-audio")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


app.include_router(vision_router, prefix="/v1/vision", tags=["vision"])
app.include_router(pipeline_router, prefix="/v1/pipeline", tags=["pipeline"])
app.include_router(pipeline_vlm_router)  # VLM experiment (prefix defined in router)
app.include_router(translation_router, prefix="/v1/translation", tags=["translation"])
app.include_router(tts_router, prefix="/v1/speech", tags=["tts"])
app.include_router(pronunciation_router, prefix="/v1/speech", tags=["pronunciation"])
