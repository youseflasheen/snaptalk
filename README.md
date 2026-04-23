# SnapTalk

SnapTalk is an interactive multimodal language-learning pipeline: upload an image, detect and segment real objects, translate target vocabulary, synthesize pronunciation audio, and optionally evaluate learner pronunciation with per-phoneme feedback.

## Overview

The active runtime flow is:

1. Object detection (LDET via YOLOv8m class-agnostic mode)
2. Object naming (Qwen2-VL or OpenAI GPT-4V provider)
3. Object segmentation (MobileSAM or bbox fallback)
4. Flashcard generation (translation + IPA + example sentence)
5. TTS generation (Edge-TTS with offline silence fallback)
6. Optional pronunciation scoring (remote/local/simulation)

The project supports:

- Interactive CLI learning flow via scripts/snap_learn.py
- Standalone pronunciation lab via scripts/pronunciation_lab.py
- FastAPI endpoints for pipeline, translation, TTS, and pronunciation

## Environment Setup

### Requirements

- Python 3.9+
- Model files in repository root:
  - yolov8m.pt
  - mobile_sam.pt

### Install

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Optional .env configuration

Create .env in repository root to override defaults from app/core/config.py:

```env
APP_NAME=SnapTalk AI Service
HOST=0.0.0.0
PORT=8000

# Translation
DEEPL_API_KEY=
GOOGLE_CLOUD_PROJECT_ID=
GOOGLE_CLOUD_CREDENTIALS_PATH=
TRANSLATION_GOOGLE_MODE=official_with_fallback

# LLM / Ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
TRANSLATION_DEFAULT_MODEL=llama3.2:3b

# Pronunciation
PRONUNCIATION_MODE=hybrid
PRONUNCIATION_LOCAL_ENABLED=false
PRONUNCIATION_MODEL_DEVICE=cpu
```

## Code Structure

```text
snaptalk/
|- app/
|  |- main.py
|  |- core/
|  |  |- config.py
|  |- routers/
|  |  |- pipeline_vlm.py
|  |  |- pronunciation.py
|  |  |- translation.py
|  |  |- tts.py
|  |- schemas/
|  |  |- pipeline.py
|  |  |- speech.py
|  |  |- translation.py
|  |  |- vision.py
|  |- services/
|  |  |- detection/
|  |  |  |- snap_learn_vlm.py
|  |  |- segmentation/
|  |  |  |- service.py
|  |  |- recognition/
|  |  |  |- vlm_providers/
|  |  |  |  |- base.py
|  |  |  |  |- openai_gpt4v.py
|  |  |  |  |- qwen2vl.py
|  |  |- translation/
|  |  |  |- service.py
|  |  |- tts/
|  |  |  |- service.py
|  |  |- pronunciation/
|  |  |  |- pronunciation_lab.py
|  |  |  |- service.py
|  |- utils/
|     |- network_security.py
|- scripts/
|  |- snap_learn.py
|  |- pronunciation_lab.py
|- docs/
|  |- RUNTIME_VERIFICATION.md
|- tests/
|- data/
|  |- seed_translations.json
|  |- yolo_world_vocab.txt
|- ARCHITECTURE_TREE.md
|- requirements.txt
```

## API Endpoints

Mounted active endpoints:

- GET /health
- POST /v1/pipeline/snap-learn-vlm
- POST /v1/translation/flashcard
- POST /v1/speech/tts
- POST /v1/speech/pronunciation

## Quick Start

### Run interactive Snap & Learn CLI

```powershell
python scripts/snap_learn.py --image "D:/path/to/image.jpg"
```

### Run standalone pronunciation lab

```powershell
python scripts/pronunciation_lab.py
```

### Run FastAPI server

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Then open:

- http://127.0.0.1:8000/docs
- http://127.0.0.1:8000/redoc

## Testing

Run all tests:

```powershell
pytest -q
```

Run selected suites:

```powershell
pytest tests/test_api.py -q
pytest tests/test_snap_learn_flow.py -q
```

## Runtime Verification

See docs/RUNTIME_VERIFICATION.md for:

- Active route topology
- Latency benchmarks
- Legacy endpoint non-exposure checks
- Translation runtime configuration checks
- Credential presence checks

## Notes

- First run may be slower due to model loading and cache warm-up.
- CPU mode is supported; CUDA improves VLM and pronunciation speed.
- Audio outputs are served from /static/audio and persisted under data/audio.

## Acknowledgement

SnapTalk combines detection, segmentation, translation, TTS, and pronunciation systems through a modular service architecture designed for practical language-learning workflows.
