#!/usr/bin/env python3
"""
Standalone Pronunciation Lab

Fast pronunciation-focused workflow that bypasses the full image pipeline.
Flow:
1) Select target language.
2) Enter translated word to practice.
3) Auto-generate reference phonemes.
4) Push-to-talk recording (press Enter to start, Enter to stop).
5) Score pronunciation with red/orange/green feedback.
6) Provide specific correction advice and retry until green (or user exits).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import unicodedata
import wave
from io import BytesIO
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import settings
from app.schemas.speech import PronunciationRequest, PronunciationResponse
from app.services.pronunciation_service import (
    score_pronunciation_from_audio_bytes,
    warmup_local_pronunciation_models,
)


LANGUAGE_OPTIONS: list[tuple[str, str]] = [
    ("ar", "Arabic"),
    ("es", "Spanish"),
    ("fr", "French"),
    ("en", "English"),
    ("de", "German"),
    ("it", "Italian"),
    ("pt", "Portuguese"),
    ("ru", "Russian"),
    ("tr", "Turkish"),
    ("nl", "Dutch"),
]

EPITRAN_LANGUAGE_MAP: dict[str, str] = {
    "ar": "ara-Arab",
    "es": "spa-Latn",
    "fr": "fra-Latn",
    "en": "eng-Latn",
    "de": "deu-Latn",
    "it": "ita-Latn",
    "pt": "por-Latn",
    "ru": "rus-Cyrl",
    "tr": "tur-Latn",
    "nl": "nld-Latn",
}

PHONEME_TIPS: dict[str, dict[str, str]] = {
    "ar": {
        "q": "Push the back of your tongue higher and deeper than a normal k sound.",
        "x": "Use a soft throat friction, like a gentle kh sound.",
        "h": "Keep a light open h breath and avoid tightening the throat.",
        "r": "Use a quick single tongue tap, not a long trill.",
        "l": "Place the tongue tip clearly on the upper gum ridge.",
    },
    "es": {
        "r": "Use a quick tongue tap for r and avoid an English r.",
        "x": "For j/x sounds, use a light throat fricative similar to h.",
        "b": "Keep lips together briefly, then release with less air than English p.",
        "d": "Touch the tongue to upper teeth lightly; avoid hard English d.",
        "a": "Keep a pure open vowel and avoid reducing it to uh.",
    },
    "fr": {
        "R": "Produce the French r from the back of the throat, not tongue-tip r.",
        "y": "Say ee while rounding your lips tightly.",
        "u": "Keep lips rounded and forward with steady airflow.",
        "e": "Use a clear closed e vowel and avoid adding glide.",
        "a": "Keep the vowel stable; avoid drifting toward uh.",
    },
}

LEVEL_RANK = {"red": 0, "orange": 1, "green": 2}

YES_TOKENS = {"y", "yes", "1", "true", "ok", "نعم", "ن", "ي", "ى"}
NO_TOKENS = {"n", "no", "0", "false", "لا", "ل"}
DEFAULT_MAX_ATTEMPTS = 7

_ENGLISH_G2P_ENGINE = None
_ENGLISH_G2P_READY = False


class Colors:
    RESET = "\033[0m"
    RED = "\033[91m"
    ORANGE = "\033[93m"
    GREEN = "\033[92m"
    CYAN = "\033[96m"


def _supports_color() -> bool:
    if os.name != "nt":
        return True
    return bool(os.getenv("ANSICON") or os.getenv("WT_SESSION"))


def _colorize_level(level: str) -> str:
    text = level.upper()
    if not _supports_color():
        return text
    if level == "green":
        return f"{Colors.GREEN}{text}{Colors.RESET}"
    if level == "orange":
        return f"{Colors.ORANGE}{text}{Colors.RESET}"
    return f"{Colors.RED}{text}{Colors.RESET}"


def _configure_console_encoding() -> None:
    # Some Windows shells use legacy encodings that cannot print IPA symbols.
    # Prefer UTF-8 and, if that is unavailable, avoid hard crashes on print.
    for stream in (sys.stdout, sys.stderr):
        if not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:
            try:
                stream.reconfigure(errors="backslashreplace")
            except Exception:
                continue


def _print_header() -> None:
    print("\n" + "=" * 68)
    print("  Standalone Pronunciation Lab")
    print("=" * 68)


def _print_section(title: str) -> None:
    print("\n" + "-" * 68)
    print(f"  {title}")
    print("-" * 68)


def _language_lookup() -> dict[str, tuple[str, str]]:
    lookup: dict[str, tuple[str, str]] = {}
    for idx, (code, name) in enumerate(LANGUAGE_OPTIONS, start=1):
        lookup[str(idx)] = (code, name)
        lookup[code.lower()] = (code, name)
        lookup[name.lower()] = (code, name)
    lookup["arabic"] = ("ar", "Arabic")
    lookup["spanish"] = ("es", "Spanish")
    lookup["french"] = ("fr", "French")
    return lookup


def _prompt_language() -> tuple[str, str]:
    _print_section("Step 1: Select Target Language")
    print("Choose a language by number, code, or name:\n")
    for idx, (code, name) in enumerate(LANGUAGE_OPTIONS, start=1):
        print(f"  [{idx:2}] {name:<10} ({code})")

    lookup = _language_lookup()
    while True:
        raw = input("\nLanguage choice: ").strip().lower()
        if raw in lookup:
            return lookup[raw]
        print("Invalid choice. Use the number, code, or full language name.")


def _prompt_word() -> str:
    _print_section("Step 2: Enter Target Word")
    while True:
        word = input("Enter the translated word you want to practice: ").strip()
        if not word:
            print("Word cannot be empty.")
            continue

        if not any(unicodedata.category(ch).startswith("L") for ch in word):
            print("Word must contain at least one letter.")
            continue

        return word


def _get_epitran_instance(lang_code: str):
    try:
        import epitran
    except Exception as exc:
        raise RuntimeError(
            f"epitran could not be imported ({type(exc).__name__})."
        ) from exc

    code = EPITRAN_LANGUAGE_MAP.get(lang_code)
    if not code:
        return None
    return epitran.Epitran(code)


def _tokenize_ipa(ipa_text: str) -> list[str]:
    phones: list[str] = []
    current = ""

    separators = {".", "|", "-", "/"}
    ignored = {"ˈ", "ˌ", "[", "]", "(", ")"}
    joiners = {"ː", "ˤ", "ʰ", "ʲ", "̩", "̯", "͡", "̃"}

    for ch in ipa_text:
        if ch.isspace() or ch in separators:
            if current:
                phones.append(current)
                current = ""
            continue

        if ch in ignored:
            continue

        if unicodedata.combining(ch) or ch in joiners:
            if current:
                current += ch
            continue

        if current:
            phones.append(current)
        current = ch

    if current:
        phones.append(current)

    return [phone for phone in phones if phone]


def _contains_arabic_script(text: str) -> bool:
    return any("\u0600" <= ch <= "\u06FF" for ch in text)


def _normalize_extracted_phonemes(
    phonemes: list[str],
    lang_code: str,
    source_word: str,
) -> tuple[list[str], str | None]:
    out: list[str] = []
    joiners = {"ː", "ˤ", "ʰ", "ʲ", "̩", "̯", "͡", "̃"}

    for token in phonemes:
        phone = token.strip()
        if not phone:
            continue

        if phone in joiners and out:
            out[-1] += phone
            continue

        if lang_code == "ar" and _contains_arabic_script(phone):
            mapped = _arabic_heuristic_phonemes(phone)
            if mapped:
                out.extend(mapped)
            continue

        out.append(phone)

    if lang_code == "ar":
        enriched, note = _enrich_arabic_reference_phonemes(source_word, out)
        return enriched, note

    return out, None


def _arabic_heuristic_phonemes(word: str) -> list[str]:
    mapping = {
        "ا": "a",
        "أ": "a",
        "إ": "i",
        "آ": "aa",
        "ب": "b",
        "ت": "t",
        "ث": "th",
        "ج": "j",
        "ح": "h",
        "خ": "kh",
        "د": "d",
        "ذ": "dh",
        "ر": "r",
        "ز": "z",
        "س": "s",
        "ش": "sh",
        "ص": "s",
        "ض": "d",
        "ط": "t",
        "ظ": "z",
        "ع": "3",
        "غ": "gh",
        "ف": "f",
        "ق": "q",
        "ك": "k",
        "ل": "l",
        "م": "m",
        "ن": "n",
        "ه": "h",
        "و": "w",
        "ي": "y",
        "ى": "a",
        "ة": "a",
        "ء": "2",
    }
    out: list[str] = []
    for ch in word:
        if ch.isspace() or ch in {"ـ", "ً", "ٌ", "ٍ", "َ", "ُ", "ِ", "ّ", "ْ"}:
            continue
        out.append(mapping.get(ch, ch))
    return out


def _is_vowel_like_phone(phone: str) -> bool:
    vowels = {
        "a",
        "i",
        "u",
        "e",
        "o",
        "aa",
        "ii",
        "uu",
        "aː",
        "iː",
        "uː",
        "ə",
        "ɐ",
        "ɛ",
        "ɪ",
        "ɔ",
        "ʊ",
        "æ",
        "ɑ",
    }
    return phone in vowels


def _arabic_letters_no_diacritics(word: str) -> list[str]:
    marks = {"ً", "ٌ", "ٍ", "َ", "ُ", "ِ", "ّ", "ْ", "ـ"}
    letters: list[str] = []
    for ch in word:
        if ch.isspace() or ch in marks:
            continue
        if "\u0600" <= ch <= "\u06FF":
            letters.append(ch)
    return letters


def _enrich_arabic_reference_phonemes(word: str, phonemes: list[str]) -> tuple[list[str], str | None]:
    """Apply lightweight Arabic fixes for undiacritized ta-marbuta noun patterns."""
    letters = _arabic_letters_no_diacritics(word)
    if len(letters) == 3 and letters[-1] == "ة" and len(phonemes) == 3:
        c1, c2, final = phonemes
        if final == "a" and not _is_vowel_like_phone(c1) and not _is_vowel_like_phone(c2):
            return [c1, "i", c2, c2, "a"], "Arabic C-C-ة enrichment"

    return phonemes, None


def _cyrillic_heuristic_phonemes(word: str) -> list[str]:
    mapping = {
        "а": "a",
        "б": "b",
        "в": "v",
        "г": "g",
        "д": "d",
        "е": "je",
        "ё": "jo",
        "ж": "zh",
        "з": "z",
        "и": "i",
        "й": "j",
        "к": "k",
        "л": "l",
        "м": "m",
        "н": "n",
        "о": "o",
        "п": "p",
        "р": "r",
        "с": "s",
        "т": "t",
        "у": "u",
        "ф": "f",
        "х": "kh",
        "ц": "ts",
        "ч": "ch",
        "ш": "sh",
        "щ": "shch",
        "ъ": "",
        "ы": "y",
        "ь": "",
        "э": "e",
        "ю": "ju",
        "я": "ja",
    }
    out: list[str] = []
    for ch in word.lower():
        if ch.isspace():
            continue
        mapped = mapping.get(ch, ch)
        if mapped:
            out.append(mapped)
    return out


def _latin_heuristic_phonemes(word: str) -> list[str]:
    text = re.sub(r"\s+", " ", word.lower()).strip()
    if not text:
        return []

    digraphs = [
        ("sch", "sk"),
        ("ch", "ch"),
        ("sh", "sh"),
        ("th", "th"),
        ("ph", "f"),
        ("ng", "ng"),
        ("qu", "kw"),
        ("ij", "ei"),
        ("oe", "u"),
        ("ee", "i"),
        ("oo", "u"),
        ("aa", "a"),
    ]
    singles = {
        "a": "a",
        "b": "b",
        "c": "k",
        "d": "d",
        "e": "e",
        "f": "f",
        "g": "g",
        "h": "h",
        "i": "i",
        "j": "dʒ",
        "k": "k",
        "l": "l",
        "m": "m",
        "n": "n",
        "o": "o",
        "p": "p",
        "q": "k",
        "r": "r",
        "s": "s",
        "t": "t",
        "u": "u",
        "v": "v",
        "w": "w",
        "x": "ks",
        "y": "j",
        "z": "z",
    }

    out: list[str] = []
    i = 0
    while i < len(text):
        if text[i] == " ":
            i += 1
            continue

        matched = False
        for pattern, phon in digraphs:
            if text.startswith(pattern, i):
                out.append(phon)
                i += len(pattern)
                matched = True
                break

        if matched:
            continue

        out.append(singles.get(text[i], text[i]))
        i += 1

    return out


def _get_english_g2p_engine():
    global _ENGLISH_G2P_ENGINE
    if _ENGLISH_G2P_ENGINE is None:
        _ensure_english_g2p_resources()
        try:
            from g2p_en import G2p
        except Exception as exc:
            raise RuntimeError(f"g2p_en import failed ({type(exc).__name__})") from exc
        _ENGLISH_G2P_ENGINE = G2p()
    return _ENGLISH_G2P_ENGINE


def _ensure_english_g2p_resources() -> None:
    global _ENGLISH_G2P_READY
    if _ENGLISH_G2P_READY:
        return

    try:
        import nltk
    except Exception as exc:
        raise RuntimeError(f"nltk import failed ({type(exc).__name__})") from exc

    required = [
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
        ("corpora/cmudict", "cmudict"),
    ]

    for resource_path, package_name in required:
        try:
            nltk.data.find(resource_path)
            continue
        except LookupError:
            pass

        nltk.download(package_name, quiet=True)

        try:
            nltk.data.find(resource_path)
        except LookupError as exc:
            raise RuntimeError(f"Missing nltk resource: {resource_path}") from exc

    _ENGLISH_G2P_READY = True


def _english_g2p_phonemes(word: str) -> list[str]:
    g2p = _get_english_g2p_engine()
    raw_tokens = g2p(word)

    phones: list[str] = []
    for token in raw_tokens:
        raw = str(token).strip()
        if not raw or raw in {" ", "'", "-", ",", ".", "!", "?", ";", ":"}:
            continue

        if re.fullmatch(r"[A-Z]{1,4}\d?", raw):
            normalized = re.sub(r"\d", "", raw).lower()
            if normalized:
                phones.append(normalized)
            continue

        if re.fullmatch(r"[A-Za-z]+", raw):
            phones.append(raw.lower())

    return phones


def _fallback_phonemes(word: str, lang_code: str) -> list[str]:
    if lang_code == "ar":
        phones = _arabic_heuristic_phonemes(word)
        enriched, _note = _enrich_arabic_reference_phonemes(word, phones)
        return enriched

    if lang_code == "ru":
        return _cyrillic_heuristic_phonemes(word)

    return _latin_heuristic_phonemes(word)


def _auto_extract_reference_phonemes(word: str, lang_code: str) -> tuple[list[str], str]:
    if lang_code == "en":
        try:
            phones = _english_g2p_phonemes(word)
            if phones:
                return phones, f"Auto phonemes from g2p_en: {' '.join(phones)}"
        except Exception as exc:
            fallback = _fallback_phonemes(word, lang_code)
            return fallback, (
                f"English g2p_en failed ({type(exc).__name__}); using heuristic fallback."
            )

        return _fallback_phonemes(word, lang_code), "English g2p_en produced empty output; using heuristic fallback."

    try:
        epi = _get_epitran_instance(lang_code)
    except Exception as exc:
        fallback = _fallback_phonemes(word, lang_code)
        return fallback, (
            f"Auto-phonemizer unavailable ({type(exc).__name__}); using heuristic phoneme fallback."
        )

    if epi is None:
        return _fallback_phonemes(word, lang_code), "No phonemizer map for this language; using heuristic fallback."

    try:
        ipa_text = str(epi.transliterate(word)).strip()
        phones = _tokenize_ipa(ipa_text)
        normalized, normalize_note = _normalize_extracted_phonemes(
            phones,
            lang_code,
            source_word=word,
        )
        if normalized:
            note_suffix = f" ({normalize_note})" if normalize_note else ""
            if normalized != phones:
                normalized_text = " ".join(normalized)
                return normalized, f"Auto phonemes from epitran: {ipa_text} -> {normalized_text}{note_suffix}"
            return normalized, f"Auto phonemes from epitran: {ipa_text}{note_suffix}"
    except Exception as exc:
        return _fallback_phonemes(word, lang_code), f"Phonemizer failed ({type(exc).__name__}); using heuristic fallback."

    return _fallback_phonemes(word, lang_code), "Phonemizer produced empty output; using heuristic fallback."


def _prompt_yes_no(prompt: str, default: bool) -> bool:
    while True:
        raw = input(prompt).strip().lower()
        if not raw:
            return default
        if raw in YES_TOKENS:
            return True
        if raw in NO_TOKENS:
            return False
        print("Please answer with y or n.")


def _confirm_reference_phonemes(phonemes: list[str]) -> list[str]:
    print(f"Reference phonemes: {' '.join(phonemes)}")
    edit = _prompt_yes_no("Edit phonemes manually? [y/N]: ", default=False)
    if not edit:
        return phonemes

    typed = input("Enter phonemes separated by spaces: ").strip()
    manual = [token for token in typed.split() if token]
    if not manual:
        print("Manual input was empty. Keeping auto phonemes.")
        return phonemes

    return manual


def _require_sounddevice():
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError("sounddevice is not installed. Run pip install -r requirements.txt") from exc
    return sd


def _record_push_to_talk(sample_rate: int, channels: int) -> np.ndarray:
    sd = _require_sounddevice()

    input("Press Enter to start recording.")
    print("Recording now. Press Enter to stop.")

    chunks: list[np.ndarray] = []

    def _callback(indata, _frames, _time, status) -> None:
        if status:
            print(f"[Audio] {status}")
        chunks.append(indata.copy())

    with sd.InputStream(
        samplerate=sample_rate,
        channels=channels,
        dtype="float32",
        callback=_callback,
    ):
        input()

    if not chunks:
        raise RuntimeError("No audio was captured. Check your microphone and try again.")

    audio = np.concatenate(chunks, axis=0)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    return np.clip(audio.astype(np.float32), -1.0, 1.0)


def _audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)

    buffer = BytesIO()
    with wave.open(buffer, "wb") as wavf:
        wavf.setnchannels(1)
        wavf.setsampwidth(2)
        wavf.setframerate(sample_rate)
        wavf.writeframes(pcm.tobytes())

    return buffer.getvalue()


def _save_attempt_audio(audio: np.ndarray, sample_rate: int, attempt: int) -> Path:
    out_dir = ROOT / "data" / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = f"pron_lab_{int(time.time())}_{attempt:02d}.wav"
    path = out_dir / filename

    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wavf:
        wavf.setnchannels(1)
        wavf.setsampwidth(2)
        wavf.setframerate(sample_rate)
        wavf.writeframes(pcm.tobytes())

    return path


def _replay_audio(audio: np.ndarray, sample_rate: int) -> None:
    sd = _require_sounddevice()
    try:
        sd.play(audio, sample_rate)
        sd.wait()
    except Exception as exc:
        print(f"Replay failed: {type(exc).__name__}")


def _phoneme_tip(lang_code: str, phoneme: str) -> str:
    language_tips = PHONEME_TIPS.get(lang_code, {})
    if phoneme in language_tips:
        return language_tips[phoneme]

    if phoneme and phoneme[0] in language_tips:
        return language_tips[phoneme[0]]

    if phoneme.lower() in {"a", "e", "i", "o", "u", "y"}:
        return "Hold a stable vowel shape and avoid reducing it to a short neutral sound."

    return "Practice this sound slowly in isolation, then blend it back into the full word."


def _print_attempt_report(resp: PronunciationResponse) -> None:
    _print_section("Scoring Result")
    print(f"Overall score: {resp.overall_score:.2f} -> {_colorize_level(resp.overall_level)}")
    print("\nPer-phoneme details:")

    for idx, item in enumerate(resp.per_phoneme, start=1):
        marker = "OK" if item.correct else "X"
        level_text = _colorize_level(item.level)
        print(
            f"  {idx:02d}. {item.phoneme:<6} score={item.score:.2f} level={level_text:<10} "
            f"status={marker} time={item.start_ms}-{item.end_ms}ms"
        )


def _print_alignment_notes(resp: PronunciationResponse) -> None:
    if resp.alignment.substitutions:
        print("\nDetected substitutions:")
        for sub in resp.alignment.substitutions[:5]:
            expected = sub.get("expected", "?")
            actual = sub.get("actual", "?")
            print(f"  - Expected '{expected}' but heard '{actual}'.")

    if resp.alignment.deletions:
        print(f"\nMissing phonemes: {' '.join(resp.alignment.deletions[:8])}")

    if resp.alignment.insertions:
        print(f"Extra phonemes: {' '.join(resp.alignment.insertions[:8])}")


def _print_correction_advice(resp: PronunciationResponse, lang_code: str) -> None:
    weak = [
        (idx, p)
        for idx, p in enumerate(resp.per_phoneme, start=1)
        if p.level in {"red", "orange"}
    ]

    if not weak and not resp.alignment.substitutions and not resp.alignment.deletions:
        print("No major issues were detected on this attempt.")
        return

    print("\nCorrection advice:")
    for idx, item in weak:
        tip = _phoneme_tip(lang_code, item.phoneme)
        print(
            f"  - Position {idx}, phoneme '{item.phoneme}': {item.level.upper()} ({item.score:.2f}). {tip}"
        )

    for sub in resp.alignment.substitutions[:4]:
        expected = sub.get("expected", "?")
        actual = sub.get("actual", "?")
        print(f"  - Replace '{actual}' with '{expected}' by slowing down that part of the word.")

    for deleted in resp.alignment.deletions[:4]:
        print(f"  - You dropped '{deleted}'. Emphasize and hold this sound slightly longer.")


def _print_improvement(previous: PronunciationResponse, current: PronunciationResponse) -> None:
    delta = round(current.overall_score - previous.overall_score, 2)
    if delta > 0:
        trend = f"Improved by +{delta:.2f}"
    elif delta < 0:
        trend = f"Dropped by {delta:.2f}"
    else:
        trend = "No overall change"

    print(f"\nAttempt comparison: {trend}")

    improved_positions: list[str] = []
    regressed_positions: list[str] = []

    size = min(len(previous.per_phoneme), len(current.per_phoneme))
    for idx in range(size):
        prev_level = previous.per_phoneme[idx].level
        curr_level = current.per_phoneme[idx].level
        if LEVEL_RANK[curr_level] > LEVEL_RANK[prev_level]:
            improved_positions.append(str(idx + 1))
        elif LEVEL_RANK[curr_level] < LEVEL_RANK[prev_level]:
            regressed_positions.append(str(idx + 1))

    if improved_positions:
        print(f"  Improved phoneme positions: {', '.join(improved_positions)}")
    if regressed_positions:
        print(f"  Regressed phoneme positions: {', '.join(regressed_positions)}")


def _configure_runtime_gpu_preferred() -> str:
    settings.pronunciation_mode = "local"
    settings.pronunciation_local_enabled = True
    settings.pronunciation_model_device = "cpu"

    try:
        import torch

        if torch.cuda.is_available():
            settings.pronunciation_model_device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            return f"GPU preferred active: using CUDA on {gpu_name}."

        return "GPU preferred active, but CUDA is not available. Falling back to CPU."
    except Exception as exc:
        return f"GPU check failed ({type(exc).__name__}). Falling back to CPU."


def _score_attempt(
    lang_code: str,
    target_word: str,
    reference_phonemes: list[str],
    wav_bytes: bytes,
) -> PronunciationResponse:
    req = PronunciationRequest(
        user_id="pron-lab-user",
        reference_text=target_word,
        reference_phonemes=reference_phonemes,
        lang_code=lang_code,
        audio_url="local-audio://attempt.wav",
    )
    return score_pronunciation_from_audio_bytes(req, wav_bytes)


def _run_word_session(
    lang_code: str,
    lang_name: str,
    sample_rate: int,
    channels: int,
    max_attempts: int,
) -> None:
    target_word = _prompt_word()
    reference_phonemes, source_note = _auto_extract_reference_phonemes(target_word, lang_code)

    _print_section("Step 3: Reference Phonemes")
    print(source_note)
    reference_phonemes = _confirm_reference_phonemes(reference_phonemes)

    print(f"\nPractice word: {target_word}")
    print(f"Language: {lang_name} ({lang_code})")
    print(f"Max attempts for this word: {max_attempts}")

    attempt = 1
    previous_response: PronunciationResponse | None = None
    best_response: PronunciationResponse | None = None

    while attempt <= max_attempts:
        _print_section(f"Attempt {attempt}")

        try:
            audio = _record_push_to_talk(sample_rate=sample_rate, channels=channels)
        except Exception as exc:
            print(f"Recording failed: {type(exc).__name__}")
            retry = input("Retry recording? [Y/n]: ").strip().lower()
            if retry == "n":
                break
            continue

        audio_path = _save_attempt_audio(audio, sample_rate=sample_rate, attempt=attempt)
        print(f"Saved attempt audio: {audio_path}")

        replay = _prompt_yes_no("Replay this recording? [y/N]: ", default=False)
        if replay:
            _replay_audio(audio, sample_rate=sample_rate)

        wav_bytes = _audio_to_wav_bytes(audio, sample_rate=sample_rate)
        response = _score_attempt(
            lang_code=lang_code,
            target_word=target_word,
            reference_phonemes=reference_phonemes,
            wav_bytes=wav_bytes,
        )

        _print_attempt_report(response)
        _print_alignment_notes(response)

        if best_response is None or response.overall_score > best_response.overall_score:
            best_response = response

        if previous_response is not None:
            _print_improvement(previous_response, response)

        if response.overall_level == "green":
            print("\nGreat work. You reached GREEN for this word.")
            break

        if attempt >= max_attempts:
            print("\nReached the maximum attempts for this word.")
            if best_response is not None:
                print(
                    "Best attempt summary: "
                    f"score={best_response.overall_score:.2f}, "
                    f"level={best_response.overall_level.upper()}"
                )
            print("Move to a new word and return later with a fresh attempt.")
            break

        _print_correction_advice(response, lang_code)

        previous_response = response
        try_again = _prompt_yes_no(
            f"\nTry again on the same word? [Y/n] ({attempt}/{max_attempts}): ",
            default=True,
        )
        if not try_again:
            break

        attempt += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone pronunciation testing lab")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Recording sample rate")
    parser.add_argument("--channels", type=int, default=1, help="Recording channels (1 recommended)")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help="Maximum attempts per word before prompting to move on",
    )
    args = parser.parse_args()

    _configure_console_encoding()
    _print_header()
    runtime_message = _configure_runtime_gpu_preferred()

    _print_section("Runtime Configuration")
    print(runtime_message)
    print(f"Pronunciation mode: {settings.pronunciation_mode}")
    print(f"Device setting: {settings.pronunciation_model_device}")
    print(
        "Thresholds: "
        f"green >= {settings.pronunciation_score_green_threshold:.2f}, "
        f"orange >= {settings.pronunciation_score_orange_threshold:.2f}, "
        "red below orange threshold"
    )

    _print_section("Model Warmup")
    warmup_ok, warmup_msg = warmup_local_pronunciation_models()
    if warmup_ok:
        print(warmup_msg)
    else:
        print(f"Warmup warning: {warmup_msg}")
        print("Scoring will continue and may use fallback behavior if local models are unavailable.")

    while True:
        lang_code, lang_name = _prompt_language()
        _run_word_session(
            lang_code=lang_code,
            lang_name=lang_name,
            sample_rate=max(8000, args.sample_rate),
            channels=max(1, args.channels),
            max_attempts=max(1, args.max_attempts),
        )

        again = _prompt_yes_no("\nPractice another word? [Y/n]: ", default=True)
        if not again:
            print("\nSession ended. Keep practicing.")
            return


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
