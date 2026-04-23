# SnapTalk Defense Master Breakdown (Latest, Evidence-Strict)

## 1) Scope and Method

This document is aligned with the current repository state after the modular refactor.

Evidence sources:
- Runtime entry and route wiring: app/main.py
- Active API routers: app/routers/pipeline_vlm.py, app/routers/translation.py, app/routers/tts.py, app/routers/pronunciation.py
- Active modular services: app/services/detection, segmentation, recognition, translation, tts, pronunciation
- Interactive flow: scripts/snap_learn.py
- Runtime checks: docs/RUNTIME_VERIFICATION.md
- Tests: tests/test_api.py, tests/test_snap_learn_flow.py
- Configuration: app/core/config.py

Rules:
- No assumptions are presented as facts.
- Legacy references are excluded unless still present in runtime behavior.

## 2) Current Runtime Entry Points

### 2.1 FastAPI Runtime

Main app behavior in app/main.py:
- Creates FastAPI app and static audio mount.
- Exposes health endpoint.
- Includes only active routers:
  - pipeline_vlm_router
  - translation_router
  - tts_router
  - pronunciation_router

Mounted active endpoints (also verified in docs/RUNTIME_VERIFICATION.md):
- GET /health
- POST /v1/pipeline/snap-learn-vlm
- POST /v1/translation/flashcard
- POST /v1/speech/tts
- POST /v1/speech/pronunciation

### 2.2 Interactive CLI Runtime

Primary flow entry:
- scripts/snap_learn.py -> main()

Key stages in CLI flow:
- Object detection via VLM pipeline service
- Translation flashcard generation
- TTS playback generation
- Optional pronunciation assessment loop

## 3) Active Modular Pipeline (Detection -> Recognition -> Segmentation -> Translation -> TTS -> Pronunciation)

### 3.1 Detection (Active)

File:
- app/services/detection/snap_learn_vlm.py

Confirmed behavior:
- Loads LDET proxy detector using YOLO with yolov8m.pt.
- Runs class-agnostic detection and duplicate suppression.
- Orchestrates downstream labeling, segmentation, and flashcard enrichment.
- Returns SnapLearnResponse.

### 3.2 Recognition (Active)

Provider package:
- app/services/recognition/vlm_providers/

Implemented providers:
- qwen2vl.py (default path)
- openai_gpt4v.py (alternate path requiring OPENAI_API_KEY)

Selection behavior:
- Provider is selected by environment variable VLM_PROVIDER.

### 3.3 Segmentation (Active)

File:
- app/services/segmentation/service.py

Confirmed behavior:
- Loads MobileSAM using mobile_sam.pt.
- Segments polygons from bounding boxes.
- Falls back to bbox polygon on failure.
- Produces transparent crops and constrained base64 PNG payloads.
- Saves artifact overlays to data/artifacts.

### 3.4 Translation (Active)

File:
- app/services/translation/service.py

Confirmed behavior:
- Public entrypoint: build_flashcard.
- Uses SQLite translation memory cache.
- Fallback chain: DeepL -> Google Cloud -> Google (deep-translator) -> MyMemory -> Ollama.
- Includes quality guards, known-term coercion, and IPA fallback/refinement.

### 3.5 TTS (Active)

File:
- app/services/tts/service.py

Confirmed behavior:
- Primary engine: Edge-TTS.
- Failure fallback: generated silence WAV.
- Bounded cleanup of old audio files.
- Returns public audio URL compatible with mounted static route.

### 3.6 Pronunciation (Active)

Files:
- app/services/pronunciation/service.py
- app/services/pronunciation/pronunciation_lab.py

Confirmed behavior:
- API scoring entrypoint: score_pronunciation.
- Local path supports Whisper + Wav2Vec components.
- Supports hybrid/local/remote/simulation modes via configuration.
- Includes per-phoneme alignment and overall level gating logic.

## 4) Model and Engine Inventory (Code-Proven)

Detection and segmentation:
- yolov8m.pt (YOLO)
- mobile_sam.pt (MobileSAM)

Recognition:
- Qwen/Qwen2-VL-2B-Instruct
- OpenAI GPT-4o (alternate provider path)

Translation engines:
- DeepL
- Google Cloud Translation API
- deep-translator Google
- deep-translator MyMemory
- Ollama chat-completions fallback

TTS:
- Edge-TTS
- Silence WAV fallback

Pronunciation:
- Whisper (default openai/whisper-tiny)
- Wav2Vec2 (default facebook/wav2vec2-base-960h)

## 5) Configuration Surface (Current)

Primary configuration file:
- app/core/config.py

Key operational settings include:
- upload/download limits for image/audio
- detector and segmentation mode defaults
- translation model and provider mode
- TTS output directory and public base URL
- pronunciation mode, model IDs, thresholds, and device

Runtime evidence summary is captured in:
- docs/RUNTIME_VERIFICATION.md

## 6) Security and Reliability Controls

Network safety utilities:
- app/utils/network_security.py

Implemented safeguards include:
- blocked private/local address validation for remote fetches
- bounded byte downloads
- content-type and scheme checks

API validations include:
- image content-type validation
- max_objects bounds validation
- upload size limit enforcement

## 7) Test Evidence Map

Core tests:
- tests/test_api.py
- tests/test_snap_learn_flow.py

Verified areas include:
- health endpoint contract
- removed legacy endpoint non-exposure (404 checks)
- oversize upload rejection
- translation cache and output quality guards
- TTS output generation behavior
- pronunciation response structure and levels
- interactive same-object multi-language loop behavior

## 8) Active Architecture Statement

Current architecture is active-modular-only:
- Detection, recognition, segmentation, translation, TTS, and pronunciation are split into dedicated service packages.
- Legacy classic pipeline/vision routes are not mounted in runtime.
- API and CLI both route through the modular service stack.

## 9) Defense Narrative (Phase-by-Phase)

1. Input and validation
- Request/image input validated for type and size.

2. Object detection
- Class-agnostic detections produced by YOLOv8m.

3. Semantic labeling
- Each object crop labeled via configurable VLM provider.

4. Segmentation and isolation
- MobileSAM polygon segmentation with bbox fallback.

5. Translation flashcard build
- Cache-first lookup with robust multi-engine fallback.

6. TTS synthesis
- Edge-TTS generation with fallback continuity path.

7. Pronunciation assessment (optional)
- Hybrid scoring path with detailed phoneme-level feedback.

## 10) Runtime Evidence Closure for Prior Gaps

The prior five gaps are now addressed by executable runtime evidence generated by:
- scripts/runtime_evidence.py
- output artifact: docs/RUNTIME_VERIFICATION.md

Resolved items:
1. Deployment topology proof:
- Runtime route audit now determines whether deployment is active-only or mixed legacy.

2. Quantitative latency proof:
- Measured benchmark table is now generated (health + translation baseline).

3. Legacy endpoint exposure proof:
- Runtime checks now record actual status codes for removed legacy endpoints.

4. Translation provider runtime truth:
- Runtime report records effective settings.translation_google_mode value loaded in process.

5. Credential presence validation:
- Runtime report now captures credential presence booleans for configured providers.

Remaining limitation (explicit):
- Credential presence is validated, but live external-provider connectivity is not guaranteed without networked integration tests.

