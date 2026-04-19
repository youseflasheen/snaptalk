# SnapTalk

SnapTalk is an interactive language-learning pipeline that turns objects from an image into multilingual flashcards with speech and pronunciation feedback.

## Main capabilities

- Detect objects from an image
- Label each object with a local VLM flow
- Translate labels into a target language
- Generate text-to-speech audio
- Run pronunciation scoring with per-phoneme feedback

## Active runtime flow

When you run scripts/snap_learn.py, the pipeline is:

1. LDET detection using yolov8m.pt
2. Qwen2-VL object labeling
3. MobileSAM segmentation using mobile_sam.pt
4. Flashcard generation from translation memory and fallback translators
5. Edge-TTS audio output
6. Optional pronunciation assessment loop

## Requirements

- Python 3.9+
- Windows, Linux, or macOS
- Internet on first run for HuggingFace model download
- Microphone if pronunciation testing is used

## Setup

1. Create and activate a virtual environment
2. Install dependencies from requirements.txt
3. Keep yolov8m.pt and mobile_sam.pt in the project root

Windows PowerShell:

python -m venv .venv
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt

## Run the interactive app

python scripts/snap_learn.py --image "D:/path/to/your/image.jpg"

You can also run with one of your local test images.

## Optional API mode

Start the API server:

uvicorn app.main:app --host 127.0.0.1 --port 8000

Health check:

GET /health

Main endpoints include:

- POST /v1/pipeline/snap-learn
- POST /v1/pipeline/snap-learn-vlm
- POST /v1/translation/flashcard
- POST /v1/speech/tts
- POST /v1/speech/pronunciation

## Runtime outputs

- data/audio stores generated and pronunciation audio
- data/artifacts stores detection and segmentation artifacts
- data/translation_memory.db stores translation memory

## Environment notes

- .env is optional but recommended for API keys and tuning
- If no premium translation key is provided, fallback translators are used

## Troubleshooting

- If model loading is slow on first run, wait for initial download completion
- If CUDA is unavailable, the pipeline falls back to CPU
- If audio playback fails, confirm OS audio device configuration
- If microphone capture fails, confirm device permissions and input selection

## Developer documentation

For full architecture and implementation workflow, see DEVELOPER_README.md
