# SnapTalk AI Integration Starter

This repository is a complete starter for cross-team integration of the SnapTalk AI pipeline.

## What this includes

- End-to-end API contracts for:
  - Vision detection + polygons
  - Translation + flashcard generation
  - TTS routing
  - Pronunciation scoring
- Runnable FastAPI backend skeleton with service wrappers
- Docker and compose setup
- Team handoff documentation for AI, Backend, and Frontend

## Why this exists

The project has one month remaining. The fastest safe strategy is contract-first integration:

1. Freeze payload shape (not model internals)
2. Build against fixed contracts in parallel
3. Keep one stable API while improving model internals safely

## Quick start

1. Create environment and install deps

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

2. Run API

```bash
uvicorn app.main:app --reload --port 8000
```

3. Open docs

- http://127.0.0.1:8000/docs

## Which pipeline is used right now?

For the command below, the project uses the **Hybrid VLM pipeline**:

```bash
python scripts/snap_learn.py --image "test1.jpeg"
```

Execution path used by this script:

1. Detection: **LDET (YOLOv8m, class-agnostic)**
2. Recognition/labeling: **Qwen2-VL-2B-Instruct** (local)
3. Segmentation: **MobileSAM** (when `SEGMENTATION=mobilesam`)
4. Translation: **DeepL first**, then Google/Microsoft/LLM fallbacks
5. TTS: **Piper first**, then **Edge-TTS** fallback if Piper voice fails

This is implemented in `scripts/snap_learn.py`, which calls `run_snap_learn_vlm` from `app/services/vlm_experiment/snap_learn_vlm.py`.

Important: the API serves both paths at the same time. The active path depends on which endpoint/script you call.

- Classic path endpoint: `POST /v1/pipeline/snap-learn`
- Hybrid VLM endpoint: `POST /v1/pipeline/snap-learn-vlm`

## Project structure

- app/: API, schemas, routers, service wrappers
- docs/: contracts and handoff guides
  - docs/ACTIVE_PIPELINE_CURRENT.md: exact active runtime pipeline and phase-by-phase breakdown

## Important notes

- This starter intentionally avoids hard-coding model weights.
- Real weights should be mounted from external storage and referenced by config.
- Production model formats should be ONNX/TensorRT/PyTorch wrappers, not .h5 for this stack.

## Vision integration

The public endpoint contract does not change while internals call real model services.

Set these environment variables:

- YOLO_SERVICE_URL
- MOBILESAM_SERVICE_URL
- RAMPP_SERVICE_URL

If a dependent model service is down, the API returns deterministic fallback objects and sets `fallback_used=true`.

## Core feature code

- Detection + ranking + segmentation + tagging pipeline: `app/services/vision_pipeline.py`
- Local ONNX model backend (YOLO-World, MobileSAM, RAM++): `app/services/vision_local.py`
- Hybrid VLM pipeline (LDET + Qwen2-VL + MobileSAM): `app/services/vlm_experiment/snap_learn_vlm.py`
- Translation memory + LLM fallback: `app/services/translation_service.py`
- Text-to-Speech generation and file serving: `app/services/tts_service.py`
- Pronunciation scoring with remote model hook + Needleman-Wunsch fallback: `app/services/pronunciation_service.py`

To run local models, set `VISION_BACKEND=local_onnx` and place model files in `models/`.

## Daily run commands

Run interactive mobile-like flow (currently Hybrid VLM path):

```bash
python scripts/snap_learn.py --image "test1.jpeg"
```

Run full automated tests:

```bash
python -m pytest -q
```

Run smoke check with real printed outputs:

```bash
python scripts/test_full_pipeline.py
```

Run complete end-to-end pipeline on a real local image:

```bash
python scripts/snap_learn.py --image "C:/path/to/your/image.jpg"
```

Run standalone pronunciation lab (fast iteration, no vision pipeline):

```bash
python scripts/pronunciation_lab.py
```
