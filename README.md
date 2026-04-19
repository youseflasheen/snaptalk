# SnapTalk

SnapTalk is an image-to-language learning app.
You give it an image, choose a target language, and it returns a learning card with:

- object name
- translation
- phonetic guide
- example sentence
- spoken audio
- optional pronunciation score

## Quick Start

Requirements:

- Python 3.9+
- yolov8m.pt in the project root
- mobile_sam.pt in the project root

Windows PowerShell setup:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run:

```powershell
python scripts/snap_learn.py --image "D:/path/to/your/image.jpg"
```

## What The Project Does (A to Z)

1. You provide an image.
2. The app detects objects in the image with YOLO (LDET-style detection).
3. For each detected object, Qwen2-VL assigns a human-readable label.
4. MobileSAM segments the object area for clean object extraction.
5. You pick your native language and target learning language.
6. The selected object is translated into the target language.
7. The app builds a flashcard with:
	- translated word
	- phonetic form
	- example sentence
8. Text-to-speech generates audio for the target word.
9. If you choose pronunciation testing:
	- the app records your voice
	- compares your pronunciation with reference phonemes
	- returns per-phoneme feedback and an overall level
10. Results are saved as runtime outputs:
	- audio files in data/audio
	- image artifacts in data/artifacts

## Notes

- First run can be slow because some models are loaded/downloaded.
- If CUDA is not available, the app runs on CPU.


## Optional API Run

If you want HTTP endpoints:

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000
```
