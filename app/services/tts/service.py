"""
Text-to-Speech Service: Edge-TTS primary with silence fallback.
"""

import asyncio
import hashlib
import logging
import os
import wave

from app.core.config import settings
from app.schemas.speech import TTSRequest, TTSResponse


EDGE_VOICES = {
    "en": "en-US-AriaNeural",
    "es": "es-ES-ElviraNeural",
    "fr": "fr-FR-DeniseNeural",
    "de": "de-DE-KatjaNeural",
    "it": "it-IT-ElsaNeural",
    "pt": "pt-BR-FranciscaNeural",
    "ja": "ja-JP-NanamiNeural",
    "ko": "ko-KR-SunHiNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
    "ar": "ar-SA-ZariyahNeural",
    "ru": "ru-RU-SvetlanaNeural",
    "tr": "tr-TR-EmelNeural",
    "nl": "nl-NL-ColetteNeural",
    "pl": "pl-PL-AgnieszkaNeural",
}

logger = logging.getLogger(__name__)


def _synthesize_edge_tts(text: str, lang_code: str, output_path: str, speed: float = 1.0) -> bool:
    """Synthesize audio using Edge-TTS cloud service."""
    try:
        import edge_tts
    except ImportError:
        logger.error("edge-tts is not installed")
        return False

    voice = EDGE_VOICES.get(lang_code, EDGE_VOICES.get("en", "en-US-AriaNeural"))
    rate_percent = int((speed - 1.0) * 100)
    rate_str = f"+{rate_percent}%" if rate_percent >= 0 else f"{rate_percent}%"

    async def _generate() -> None:
        communicate = edge_tts.Communicate(text, voice, rate=rate_str)
        await communicate.save(output_path)

    try:
        asyncio.run(_generate())
        return True
    except RuntimeError:
        # Fallback for environments where an event loop is already active.
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_generate())
            return True
        except Exception as exc:
            logger.error("Edge-TTS synthesis failed: %s", exc)
            return False
        finally:
            loop.close()
    except Exception as exc:
        logger.error("Edge-TTS synthesis failed: %s", exc)
        return False


def _save_silence_wav(path: str, duration_ms: int, sample_rate: int = 22050) -> None:
    n_frames = int(sample_rate * (duration_ms / 1000.0))
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * n_frames)


def _tts_cache_key(req: TTSRequest) -> str:
    payload = f"{req.text}|{req.lang_code.lower()}|{req.voice}|{req.speed:.2f}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _cleanup_old_audio(output_dir: str, keep_file: str, max_files: int = 5) -> None:
    try:
        audio_files = []
        for file_name in os.listdir(output_dir):
            if file_name.endswith(".wav") and file_name.startswith("aud_"):
                full_path = os.path.join(output_dir, file_name)
                audio_files.append((full_path, os.path.getmtime(full_path)))

        audio_files.sort(key=lambda item: item[1])
        if len(audio_files) > max_files:
            for file_path, _ in audio_files[:-max_files]:
                if os.path.basename(file_path) != keep_file:
                    try:
                        os.remove(file_path)
                    except Exception:
                        pass
    except Exception:
        pass


def synthesize(req: TTSRequest, cleanup: bool = True) -> TTSResponse:
    """Synthesize speech with Edge-TTS, then fallback to silence if needed."""
    normalized_lang = req.lang_code.strip().lower()[:2]

    audio_id = f"aud_{_tts_cache_key(req)}"
    base_duration = max(300, int(len(req.text) * 55))
    duration_ms = max(220, int(base_duration / req.speed))

    output_dir = settings.tts_output_dir
    os.makedirs(output_dir, exist_ok=True)

    file_name = f"{audio_id}.wav"
    local_path = os.path.join(output_dir, file_name)

    engine_used = "edge"

    if not os.path.exists(local_path):
        success = _synthesize_edge_tts(req.text, normalized_lang, local_path, req.speed)
        if not success:
            engine_used = "silence"
            _save_silence_wav(local_path, duration_ms)

    if cleanup:
        _cleanup_old_audio(output_dir, file_name, max_files=5)

    base_url = settings.public_base_url.rstrip("/")

    return TTSResponse(
        audio_id=audio_id,
        engine=engine_used,
        audio_url=f"{base_url}/static/audio/{file_name}",
        duration_ms=duration_ms,
    )
