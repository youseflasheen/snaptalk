from __future__ import annotations

import difflib
import logging
import random
import threading
import wave
from io import BytesIO
from typing import Optional

import httpx
import numpy as np

from app.core.config import settings
from app.schemas.speech import AlignmentResult, PhonemeResult, PronunciationRequest, PronunciationResponse
from app.utils.network_security import download_bytes_with_limit, parse_allowed_hosts

logger = logging.getLogger(__name__)

_whisper_pipeline = None
_whisper_lock = threading.Lock()

_wav2vec_processor = None
_wav2vec_model = None
_wav2vec_device = "cpu"
_wav2vec_lock = threading.Lock()


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _score_to_level(score: float) -> str:
    if score >= settings.pronunciation_score_green_threshold:
        return "green"
    if score >= settings.pronunciation_score_orange_threshold:
        return "orange"
    return "red"


def _resolve_overall_level(overall_score: float, phoneme_levels: list[str]) -> str:
    """Gate GREEN overall scores with bounded tolerance for isolated weak phonemes."""
    base_level = _score_to_level(overall_score)
    if base_level != "green":
        return base_level

    total = len(phoneme_levels)
    if total == 0:
        return base_level

    red_count = phoneme_levels.count("red")
    orange_count = phoneme_levels.count("orange")

    if red_count >= max(2, total // 2):
        return "red"

    if red_count > 0:
        return "orange"

    if orange_count >= max(2, (total + 2) // 3):
        return "orange"

    if orange_count == 1 and overall_score < 0.90:
        return "orange"

    return "green"


def _simulate_observed_phonemes(reference: list[str], seed: int) -> list[str]:
    """Deterministic simulation from audio-derived seed until ASR model is wired."""
    rng = random.Random(seed)
    out: list[str] = []

    for ph in reference:
        roll = rng.random()
        if roll < 0.12:
            # deletion
            continue
        if roll < 0.24:
            # substitution
            out.append("l" if ph != "l" else "r")
            continue
        out.append(ph)
        if rng.random() < 0.08:
            # insertion
            out.append("uh")

    return out or reference[:]


def _perturb_reference_phonemes(reference: list[str], seed: int, intensity: float) -> list[str]:
    """Perturb reference phonemes using intensity derived from lexical mismatch."""
    rng = random.Random(seed)
    intensity = _clamp(intensity)

    deletion_rate = 0.04 + (0.20 * intensity)
    substitution_rate = 0.06 + (0.25 * intensity)
    insertion_rate = 0.03 + (0.12 * intensity)

    out: list[str] = []
    for ph in reference:
        roll = rng.random()
        if roll < deletion_rate:
            continue
        if roll < deletion_rate + substitution_rate:
            out.append("l" if ph != "l" else "r")
        else:
            out.append(ph)
        if rng.random() < insertion_rate:
            out.append("uh")

    return out or reference[:]


def _seed_from_bytes(data: bytes) -> int:
    return int(sum(data[:2048]) + len(data))


def _download_audio_bytes(audio_url: str) -> bytes:
    return download_bytes_with_limit(
        audio_url,
        timeout_seconds=settings.pronunciation_model_timeout_seconds,
        max_bytes=settings.max_audio_fetch_bytes,
        allowed_hosts=parse_allowed_hosts(settings.allowed_external_hosts),
        allow_private=settings.allow_private_network_urls,
        accepted_content_prefixes=("audio/", "application/octet-stream"),
    )


def _decode_wav_to_mono_16k(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    with wave.open(BytesIO(audio_bytes), "rb") as wavf:
        channels = wavf.getnchannels()
        sample_width = wavf.getsampwidth()
        sample_rate = wavf.getframerate()
        raw_frames = wavf.readframes(wavf.getnframes())

    if sample_width == 1:
        samples = (np.frombuffer(raw_frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        samples = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        samples = np.frombuffer(raw_frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError("Unsupported WAV sample width")

    if channels > 1:
        frame_count = len(samples) // channels
        samples = samples[: frame_count * channels].reshape(frame_count, channels).mean(axis=1)

    samples = np.clip(samples, -1.0, 1.0)
    if sample_rate != 16000 and samples.size > 1:
        target_count = max(1, int(round((samples.size * 16000) / sample_rate)))
        src_index = np.arange(samples.size, dtype=np.float32)
        tgt_index = np.linspace(0, samples.size - 1, target_count, dtype=np.float32)
        samples = np.interp(tgt_index, src_index, samples).astype(np.float32)

    duration_ms = int((samples.size / 16000.0) * 1000)
    return samples.astype(np.float32), max(duration_ms, 1)


def _audio_seed(audio_url: str) -> int:
    try:
        data = _download_audio_bytes(audio_url)
    except Exception:
        data = audio_url.encode("utf-8")

    # If valid wav, include frame count in seed for better stability.
    try:
        with wave.open(BytesIO(data), "rb") as wavf:
            frames = wavf.getnframes()
            rate = wavf.getframerate()
            return int(frames + rate + len(data))
    except Exception:
        return int(sum(data[:2048]) + len(data))


def _needleman_wunsch(ref: list[str], hyp: list[str]) -> tuple[list[str], list[str]]:
    match_score = 2
    mismatch_penalty = -1
    gap_penalty = -1

    n = len(ref)
    m = len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bt = [[""] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i * gap_penalty
        bt[i][0] = "U"
    for j in range(1, m + 1):
        dp[0][j] = j * gap_penalty
        bt[0][j] = "L"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = dp[i - 1][j - 1] + (match_score if ref[i - 1] == hyp[j - 1] else mismatch_penalty)
            up = dp[i - 1][j] + gap_penalty
            left = dp[i][j - 1] + gap_penalty
            best = max(diag, up, left)
            dp[i][j] = best
            bt[i][j] = "D" if best == diag else ("U" if best == up else "L")

    aligned_ref: list[str] = []
    aligned_hyp: list[str] = []
    i, j = n, m
    while i > 0 or j > 0:
        direction = bt[i][j] if i >= 0 and j >= 0 else ""
        if i > 0 and j > 0 and direction == "D":
            aligned_ref.append(ref[i - 1])
            aligned_hyp.append(hyp[j - 1])
            i -= 1
            j -= 1
        elif i > 0 and direction == "U":
            aligned_ref.append(ref[i - 1])
            aligned_hyp.append("-")
            i -= 1
        else:
            aligned_ref.append("-")
            aligned_hyp.append(hyp[j - 1])
            j -= 1

    aligned_ref.reverse()
    aligned_hyp.reverse()
    return aligned_ref, aligned_hyp


def _normalize_for_compare(text: str) -> str:
    return "".join(ch.lower() for ch in text if ch.isalnum())


def _lexical_similarity(reference_text: str, observed_text: str) -> float:
    ref = _normalize_for_compare(reference_text)
    hyp = _normalize_for_compare(observed_text)
    if not ref or not hyp:
        return 0.0
    return _clamp(difflib.SequenceMatcher(a=ref, b=hyp).ratio())


def _select_whisper_language(lang_code: str) -> str:
    mapping = {
        "en": "english",
        "es": "spanish",
        "fr": "french",
        "de": "german",
        "it": "italian",
        "pt": "portuguese",
        "ja": "japanese",
        "ko": "korean",
        "ar": "arabic",
        "zh": "chinese",
        "ru": "russian",
        "tr": "turkish",
        "nl": "dutch",
    }
    return mapping.get(lang_code.strip().lower()[:2], "english")


def _get_whisper_pipeline():
    global _whisper_pipeline
    if _whisper_pipeline is None:
        with _whisper_lock:
            if _whisper_pipeline is None:
                from transformers import pipeline

                whisper_device = -1
                if settings.pronunciation_model_device.lower() == "cuda":
                    try:
                        import torch

                        if torch.cuda.is_available():
                            whisper_device = 0
                    except Exception:
                        whisper_device = -1

                _whisper_pipeline = pipeline(
                    "automatic-speech-recognition",
                    model=settings.pronunciation_whisper_model,
                    device=whisper_device,
                )
    return _whisper_pipeline


def _transcribe_with_whisper(samples: np.ndarray, lang_code: str) -> str:
    pipe = _get_whisper_pipeline()
    payload = {"array": samples, "sampling_rate": 16000}
    language = _select_whisper_language(lang_code)
    try:
        result = pipe(payload, generate_kwargs={"task": "transcribe", "language": language})
    except TypeError:
        result = pipe(payload)
    if isinstance(result, dict):
        return str(result.get("text", "")).strip()
    return str(result).strip()


def _get_wav2vec_bundle():
    global _wav2vec_model, _wav2vec_processor, _wav2vec_device

    if _wav2vec_model is None or _wav2vec_processor is None:
        with _wav2vec_lock:
            if _wav2vec_model is None or _wav2vec_processor is None:
                import torch
                from transformers import AutoModelForCTC, AutoProcessor

                _wav2vec_processor = AutoProcessor.from_pretrained(settings.pronunciation_wav2vec_model)
                _wav2vec_model = AutoModelForCTC.from_pretrained(settings.pronunciation_wav2vec_model)

                if settings.pronunciation_model_device.lower() == "cuda" and torch.cuda.is_available():
                    _wav2vec_device = "cuda"
                    _wav2vec_model = _wav2vec_model.to("cuda")
                else:
                    _wav2vec_device = "cpu"

                _wav2vec_model.eval()

    return _wav2vec_processor, _wav2vec_model, _wav2vec_device


def _wav2vec_margins(samples: np.ndarray) -> tuple[list[float], str]:
    try:
        import torch

        processor, model, model_device = _get_wav2vec_bundle()

        encoded = processor(samples, sampling_rate=16000, return_tensors="pt", padding=True)
        if model_device == "cuda":
            encoded = {key: value.to("cuda") for key, value in encoded.items()}

        with torch.no_grad():
            logits = model(**encoded).logits[0]

        probs = torch.softmax(logits, dim=-1)
        top2 = torch.topk(probs, k=2, dim=-1).values
        margins = (top2[:, 0] - top2[:, 1]).detach().cpu().numpy().astype(np.float32)

        token_ids = torch.argmax(logits, dim=-1)
        decoded = processor.batch_decode(token_ids.unsqueeze(0))[0]
        return margins.tolist(), str(decoded).strip()
    except Exception:
        logger.exception("Local Wav2Vec scoring failed")
        return [], ""


def _margin_for_phoneme(frame_margins: list[float], phoneme_index: int, total_phonemes: int) -> float:
    if not frame_margins:
        return 0.6
    if total_phonemes <= 0:
        return 0.6

    frame_count = len(frame_margins)
    start = int((phoneme_index / total_phonemes) * frame_count)
    end = int(((phoneme_index + 1) / total_phonemes) * frame_count)
    start = max(0, min(start, frame_count - 1))
    end = max(start + 1, min(end, frame_count))

    segment = frame_margins[start:end]
    if not segment:
        return 0.6

    return _clamp(float(sum(segment) / len(segment)))


def _average_margin(frame_margins: list[float]) -> float:
    if not frame_margins:
        return 0.6
    return _clamp(float(sum(frame_margins) / len(frame_margins)))


def _build_pronunciation_response(
    reference_phonemes: list[str],
    observed_phonemes: list[str],
    total_duration_ms: int,
    lexical_score: float,
    frame_margins: list[float],
) -> PronunciationResponse:
    aligned_ref, aligned_hyp = _needleman_wunsch(reference_phonemes, observed_phonemes)

    per_phoneme: list[PhonemeResult] = []
    insertions: list[str] = []
    deletions: list[str] = []
    substitutions: list[dict[str, str]] = []

    total_ref = max(len(reference_phonemes), 1)
    step_ms = max(45, int(total_duration_ms / total_ref))
    t = 0
    ref_index = 0

    for r, h in zip(aligned_ref, aligned_hyp):
        if r == "-":
            insertions.append(h)
            continue

        acoustic = _margin_for_phoneme(frame_margins, ref_index, total_ref)
        ref_index += 1

        if h == "-":
            deletions.append(r)
            raw_score = 0.32 + (0.30 * acoustic)
            lexical_weight = 0.78 + (0.22 * lexical_score)
            correct = False
        elif r == h:
            # GOP-like score: high posterior margins increase confidence on matched phonemes.
            raw_score = 0.58 + (0.42 * acoustic)
            lexical_weight = 0.88 + (0.12 * lexical_score)
            correct = True
        else:
            substitutions.append({"expected": r, "actual": h})
            raw_score = 0.36 + (0.34 * acoustic)
            lexical_weight = 0.78 + (0.22 * lexical_score)
            correct = False

        score = _clamp(raw_score * lexical_weight)
        level = _score_to_level(score)
        per_phoneme.append(
            PhonemeResult(
                phoneme=r,
                correct=correct,
                score=round(score, 2),
                level=level,
                start_ms=t,
                end_ms=min(total_duration_ms, t + step_ms),
            )
        )
        t += step_ms

    overall_score = round(sum(p.score for p in per_phoneme) / max(len(per_phoneme), 1), 2)
    overall_level = _resolve_overall_level(overall_score, [p.level for p in per_phoneme])

    return PronunciationResponse(
        overall_score=overall_score,
        overall_level=overall_level,
        per_phoneme=per_phoneme,
        alignment=AlignmentResult(
            insertions=insertions,
            deletions=deletions,
            substitutions=substitutions,
        ),
    )


def _score_pronunciation_local_from_audio_bytes(
    req: PronunciationRequest,
    audio_bytes: bytes,
) -> Optional[PronunciationResponse]:
    if not settings.pronunciation_local_enabled:
        return None

    try:
        samples, duration_ms = _decode_wav_to_mono_16k(audio_bytes)
    except Exception:
        logger.exception("Unable to prepare audio for local pronunciation scoring")
        return None

    if samples.size == 0:
        return None

    transcript = ""
    lexical_score = 0.0
    try:
        transcript = _transcribe_with_whisper(samples, req.lang_code)
        lexical_score = _lexical_similarity(req.reference_text, transcript)
    except Exception:
        logger.exception("Local Whisper transcription failed")

    frame_margins, wav2vec_text = _wav2vec_margins(samples)
    if not transcript and wav2vec_text:
        transcript = wav2vec_text
        lexical_score = _lexical_similarity(req.reference_text, transcript)

    acoustic_signal = _average_margin(frame_margins)
    combined_confidence = _clamp((0.70 * lexical_score) + (0.30 * acoustic_signal))

    seed = _seed_from_bytes(audio_bytes + transcript.encode("utf-8"))
    if combined_confidence >= 0.74:
        observed = req.reference_phonemes[:]
    elif combined_confidence >= 0.38:
        observed = _perturb_reference_phonemes(
            req.reference_phonemes,
            seed=seed,
            intensity=1.0 - combined_confidence,
        )
    else:
        observed = _simulate_observed_phonemes(req.reference_phonemes, seed)

    return _build_pronunciation_response(
        reference_phonemes=req.reference_phonemes,
        observed_phonemes=observed,
        total_duration_ms=duration_ms,
        lexical_score=lexical_score,
        frame_margins=frame_margins,
    )


def _score_pronunciation_local(req: PronunciationRequest) -> Optional[PronunciationResponse]:
    if not settings.pronunciation_local_enabled:
        return None

    try:
        audio_bytes = _download_audio_bytes(req.audio_url)
    except Exception:
        logger.exception("Unable to download audio for local pronunciation scoring")
        return None

    return _score_pronunciation_local_from_audio_bytes(req, audio_bytes)


def _score_pronunciation_simulation(req: PronunciationRequest) -> PronunciationResponse:
    seed = _audio_seed(req.audio_url)
    observed = _simulate_observed_phonemes(req.reference_phonemes, seed)
    duration_ms = max(400, len(req.reference_phonemes) * 95)
    return _build_pronunciation_response(
        reference_phonemes=req.reference_phonemes,
        observed_phonemes=observed,
        total_duration_ms=duration_ms,
        lexical_score=0.68,
        frame_margins=[],
    )


def score_pronunciation(req: PronunciationRequest) -> PronunciationResponse:
    mode = settings.pronunciation_mode.strip().lower()

    if mode in {"remote", "hybrid"}:
        remote = _score_pronunciation_remote(req)
        if remote is not None:
            return remote

    if mode in {"local", "hybrid"}:
        local = _score_pronunciation_local(req)
        if local is not None:
            return local

    return _score_pronunciation_simulation(req)


def score_pronunciation_from_audio_bytes(
    req: PronunciationRequest,
    audio_bytes: bytes,
) -> PronunciationResponse:
    """Score pronunciation from in-memory audio bytes for standalone/local workflows."""
    mode = settings.pronunciation_mode.strip().lower()

    if mode in {"local", "hybrid"}:
        local = _score_pronunciation_local_from_audio_bytes(req, audio_bytes)
        if local is not None:
            return local

    return _score_pronunciation_simulation(req)


def warmup_local_pronunciation_models() -> tuple[bool, str]:
    """Preload local pronunciation models to reduce first-attempt latency spikes."""
    if not settings.pronunciation_local_enabled:
        return False, "Local pronunciation is disabled."

    try:
        _get_whisper_pipeline()
        _, _, model_device = _get_wav2vec_bundle()
        return True, f"Local models loaded on {model_device}."
    except Exception as exc:
        logger.exception("Pronunciation model warmup failed")
        return False, f"Warmup failed: {type(exc).__name__}"


def _score_pronunciation_remote(req: PronunciationRequest) -> Optional[PronunciationResponse]:
    payload = {
        "user_id": req.user_id,
        "reference_text": req.reference_text,
        "reference_phonemes": req.reference_phonemes,
        "lang_code": req.lang_code,
        "audio_url": req.audio_url,
    }
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.post(settings.pronunciation_service_url, json=payload)
            resp.raise_for_status()
            return PronunciationResponse(**resp.json())
    except Exception:
        return None
