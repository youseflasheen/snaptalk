# SnapTalk Developer Guide

This document explains SnapTalk from architecture to runtime behavior so developers can maintain and extend the project safely.

## 1. Project purpose

SnapTalk converts detected image objects into learning artifacts:

- target-language word
- phonetic guidance
- example sentence
- generated speech audio
- optional pronunciation quality feedback

## 2. High-level architecture

Core layers:

1. Entrypoints
- scripts/snap_learn.py for interactive flow
- app/main.py for API flow

2. Routers
- app/routers/*.py expose HTTP endpoints

3. Services
- app/services/vlm_experiment/snap_learn_vlm.py for active hybrid vision path
- app/services/translation_service.py for translation and flashcard building
- app/services/tts_service.py for speech synthesis
- app/services/pronunciation_service.py for pronunciation scoring
- app/services/snap_learn_service.py for shared segmentation and artifact helpers

4. Schemas
- app/schemas/*.py for request and response contracts

5. Configuration
- app/core/config.py centralizes environment settings

## 3. Active pipeline internals

Current script runtime is hybrid local vision:

A. Detection
- LDET proxy uses YOLO class-agnostic detection from yolov8m.pt

B. Labeling
- Qwen2-VL provider identifies object crops

C. Segmentation
- MobileSAM segments each selected box using mobile_sam.pt

D. Flashcard generation
- build_flashcard resolves translation with cache-first strategy

E. Audio generation
- Edge-TTS synthesizes target word audio

F. Pronunciation loop
- optional recording and scoring with per-phoneme diagnostics

## 4. Translation workflow details

Translation flow in app/services/translation_service.py:

1. normalize source token
2. lookup in SQLite translation memory
3. if missing, call fallback chain
4. generate IPA and examples
5. persist verified result to translation memory

Storage:
- data/translation_memory.db
- optional seed data in data/seed_translations.json

## 5. Pronunciation workflow details

Interactive path uses scripts/pronunciation_lab.py and pronunciation service:

1. create or select reference phonemes
2. record microphone attempt
3. score acoustic and alignment quality
4. compute overall level (red or orange or green)
5. provide correction hints
6. iterate attempts until target or exit

## 6. API workflow details

Main server file:
- app/main.py

Mounted routes:
- pipeline routes for object-learning flow
- translation route for flashcard requests
- tts route for speech requests
- pronunciation route for scoring requests
- vision route for external URL vision requests

## 7. Data and generated assets

Runtime directories:
- data/audio for wav outputs
- data/artifacts for segmented outputs and overlays

Generated files are intentionally ignored by git.

## 8. Configuration model

Configuration lives in app/core/config.py using pydantic settings.

Important groups:
- translation settings
- tts settings
- pronunciation settings
- vision backend settings
- network safety settings

## 9. Safety and network controls

Network helper file:
- app/utils/network_security.py

Features:
- host allowlist parsing
- private network protection
- URL validation
- download size limiting

## 10. Typical development workflow

1. install dependencies
2. run script mode for integration validation
3. run API mode for endpoint validation
4. inspect data/artifacts and data/audio outputs
5. tune settings through .env
6. update schemas first when changing API contracts
7. update service logic and then router wiring
8. regression test script and API paths before commit

## 11. Extension points

Common extension tasks:

- add language support in translation and tts voice maps
- improve labeling quality in qwen2vl provider
- refine pronunciation scoring thresholds and alignment heuristics
- add post-processing in snap_learn_vlm response assembly

## 12. File map

Primary files for fast onboarding:

- scripts/snap_learn.py
- scripts/pronunciation_lab.py
- app/main.py
- app/core/config.py
- app/services/vlm_experiment/snap_learn_vlm.py
- app/services/translation_service.py
- app/services/tts_service.py
- app/services/pronunciation_service.py
- app/services/snap_learn_service.py
- app/utils/network_security.py

## 13. Known runtime behaviors

- first run may be slow due to model download and cache warmup
- CPU fallback is expected when CUDA is unavailable
- translation quality depends on available keys and fallback engine path
- pronunciation quality depends on microphone input quality and language phoneme mapping

## 14. Release checklist

Before push:

1. confirm script mode runs successfully with a real image
2. confirm API starts and health endpoint returns ok
3. ensure generated runtime artifacts are not staged
4. keep large model binaries outside git
5. verify README and DEVELOPER_README remain synchronized with behavior
