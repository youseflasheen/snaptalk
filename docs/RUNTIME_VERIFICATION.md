# Runtime Verification

Generated at: 2026-04-23T07:09:33

## 1) Deployment Topology (Runtime-Proven)

Resolved topology mode: active-modular-only
Legacy routes mounted: no

Mounted routes:

| Methods | Path |
|---|---|
| GET | /docs |
| GET | /docs/oauth2-redirect |
| GET | /health |
| GET | /openapi.json |
| GET | /redoc |
| POST | /v1/pipeline/snap-learn-vlm |
| POST | /v1/speech/pronunciation |
| POST | /v1/speech/tts |
| POST | /v1/translation/flashcard |

## 2) Quantitative Latency Benchmarks

Benchmarks executed in-process with FastAPI TestClient.

| Endpoint | Loops | Status codes | Min (ms) | P50 (ms) | P95 (ms) | Max (ms) |
|---|---:|---|---:|---:|---:|---:|
| GET /health | 15 | [200] | 12.68 | 13.3 | 14.16 | 33.15 |
| POST /v1/translation/flashcard (en->en) | 15 | [200] | 18.07 | 20.47 | 23.29 | 31.16 |

## 3) Legacy Endpoint Exposure Check

POST /v1/vision/detect status: 404
POST /v1/pipeline/snap-learn status: 404

## 4) Translation Runtime Configuration

settings.translation_google_mode: deep_translator_only
settings.translation_default_model: llama3.2:3b
en->en flashcard source observed: translation_memory (status 200)

## 5) Credential Presence Validation

This verifies presence only, not external provider connectivity.

| Credential/Input | Present |
|---|---|
| DeepL API key configured | yes |
| Google Cloud project id configured | yes |
| Google credentials path configured | yes |
| Google credentials file exists at configured path | no |
