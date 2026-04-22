#!/usr/bin/env python3
"""
SnapTalk Interactive Pipeline
==============================

Complete interactive terminal interface for the full learning flow:
1. Load image and detect objects.
2. Prompt once for native language and initial target language.
3. Show detected object label translated to the native language.
4. Generate flashcard in target language (word + phonetic guide + examples).
5. Auto-play target-word TTS audio.
6. Optional pronunciation assessment with retry feedback loop.
7. Optional multi-language rerun for the same object.

Usage:
    python scripts/snap_learn.py --image "path/to/image.jpg"
    python scripts/snap_learn.py  # Uses default test image

Detection Model: LDET (YOLOv8m in class-agnostic mode)
Labeling Model: Qwen2-VL-2B (GPU accelerated)
Segmentation: MobileSAM
Translation: SQLite cache -> DeepL -> Google Cloud -> Google -> MyMemory -> Ollama
TTS: Edge-TTS (cloud) with silence fallback

Human Verification:
    New translations are verified by the user before being saved.
    This ensures 100% accuracy for future lookups from the database.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import settings

# Set environment for detection
os.environ.setdefault("DETECTOR", "ldet")
os.environ.setdefault("SEGMENTATION", "mobilesam")

# Supported languages with display names
SUPPORTED_LANGUAGES = {
    "en": "English",
    "es": "Spanish (Español)",
    "fr": "French (Français)",
    "ja": "Japanese (日本語)",
    "de": "German (Deutsch)",
    "it": "Italian (Italiano)",
    "pt": "Portuguese (Português)",
    "ar": "Arabic (العربية)",
    "zh": "Chinese (中文)",
    "ko": "Korean (한국어)",
    "ru": "Russian (Русский)",
    "nl": "Dutch (Nederlands)",
    "tr": "Turkish (Türkçe)",
}

TRANSLATOR_LANG_MAP = {
    "en": "en",
    "es": "es",
    "fr": "fr",
    "ja": "ja",
    "de": "de",
    "it": "it",
    "pt": "pt",
    "ar": "ar",
    "zh": "zh-CN",
    "ko": "ko",
    "ru": "ru",
    "nl": "nl",
    "tr": "tr",
}

_PRONUNCIATION_READY = False
_SESSION_NATIVE_LANG = "en"
_UI_TRANSLATION_CACHE: dict[tuple[str, str], str] = {}
_NATIVE_LABEL_CACHE: dict[str, str] = {}
_SESSION_LAST_TTS_AUDIO_PATH: str | None = None
_SESSION_LAST_PRON_AUDIO_PATH: str | None = None

_STATIC_UI_TRANSLATIONS: dict[str, dict[str, str]] = {
    "ja": {
        "Select an object to translate:": "翻訳するオブジェクトを選択してください:",
        "Object Selection": "オブジェクト選択",
        "Enter object number:": "オブジェクト番号を入力してください:",
        "Please enter a number.": "数字を入力してください。",
        "Available languages:": "利用可能な言語:",
        "Enter language number:": "言語番号を入力してください:",
        "Detected object": "検出されたオブジェクト",
        "Target language": "学習対象言語",
        "Target word": "学習単語",
        "Phonetic guide": "発音ガイド",
        "Phonetic source": "発音ソース",
        "Do you want to test your pronunciation for this word?": "この単語の発音テストをしますか?",
        "Do you want to translate this word into another language?": "この単語を別の言語にも翻訳しますか?",
        "Do you want to select another detected object from this image?": "この画像から別の検出オブジェクトを選択しますか?",
        "Please answer with y or n.": "y または n で回答してください。",
        "Thanks for using SnapTalk! Goodbye!": "SnapTalkをご利用いただきありがとうございました。",
        "Audio files": "音声ファイル",
    }
}


def _set_session_native_language(lang_code: str) -> None:
    global _SESSION_NATIVE_LANG
    _SESSION_NATIVE_LANG = (lang_code or "en").strip().lower()
    _UI_TRANSLATION_CACHE.clear()
    _NATIVE_LABEL_CACHE.clear()


def _ui(text: str) -> str:
    lang = _SESSION_NATIVE_LANG
    if lang == "en" or not text.strip():
        return text

    static_map = _STATIC_UI_TRANSLATIONS.get(lang, {})
    if text in static_map:
        return static_map[text]

    cache_key = (lang, text)
    if cache_key in _UI_TRANSLATION_CACHE:
        return _UI_TRANSLATION_CACHE[cache_key]

    translated = _translate_english_text_to_language(text, lang)
    resolved = translated.strip() if translated and translated.strip() else text
    _UI_TRANSLATION_CACHE[cache_key] = resolved
    return resolved


def _audio_dir() -> Path:
    return Path(settings.tts_output_dir).resolve()


def _prune_audio_storage(keep_paths: list[str]) -> None:
    audio_dir = _audio_dir()
    if not audio_dir.exists():
        return

    keep: set[str] = set()
    for path in keep_paths:
        if path:
            keep.add(str(Path(path).resolve()))

    for wav_path in audio_dir.glob("*.wav"):
        abs_path = str(wav_path.resolve())
        if abs_path in keep:
            continue
        try:
            wav_path.unlink()
        except Exception:
            continue


def _start_audio_session() -> None:
    global _SESSION_LAST_TTS_AUDIO_PATH, _SESSION_LAST_PRON_AUDIO_PATH
    _SESSION_LAST_TTS_AUDIO_PATH = None
    _SESSION_LAST_PRON_AUDIO_PATH = None

    audio_dir = _audio_dir()
    audio_dir.mkdir(parents=True, exist_ok=True)
    _prune_audio_storage([])


def _native_label_for_object(source_word: str) -> str:
    if _SESSION_NATIVE_LANG == "en":
        return source_word

    if source_word in _NATIVE_LABEL_CACHE:
        return _NATIVE_LABEL_CACHE[source_word]

    translated = translate_label_to_native(source_word, _SESSION_NATIVE_LANG)
    _NATIVE_LABEL_CACHE[source_word] = translated
    return translated


def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """Print application header."""
    print("\n" + "=" * 60)
    print("  📸 SNAPTALK - Snap & Learn Interactive Pipeline")
    print("=" * 60)


def print_section(title: str):
    """Print section header."""
    localized = _ui(title)
    print(f"\n{'─' * 60}")
    print(f"  {localized}")
    print(f"{'─' * 60}")


def _language_name(lang_code: str) -> str:
    return SUPPORTED_LANGUAGES.get(lang_code, lang_code).split(" (")[0]


def _prompt_yes_no(prompt: str, default: bool) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        raw = input(f"{_ui(prompt)} {suffix}: ").strip().lower()
        if not raw:
            return default
        if raw in {"y", "yes", "1", "true"}:
            return True
        if raw in {"n", "no", "0", "false"}:
            return False
        print(f"  ❌ {_ui('Please answer with y or n.')}")


def load_image(image_path: str) -> bytes:
    """Load image from file."""
    print_section("📷 STEP 1: Loading Image")
    
    if not os.path.exists(image_path):
        print(f"  ❌ Image not found: {image_path}")
        sys.exit(1)
    
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    size_kb = len(image_bytes) / 1024
    print(f"  ✅ Loaded: {os.path.basename(image_path)} ({size_kb:.1f} KB)")
    return image_bytes


def detect_objects(image_bytes: bytes, max_objects: int = 10):
    """Run object detection and return detected objects."""
    print_section("🔍 STEP 1B: Detecting Objects")
    print("  ⏳ Running LDET detection + Qwen2-VL labeling...")
    print("     (This may take 30-60 seconds on first run)")
    
    from app.services.detection.snap_learn_vlm import run_snap_learn_vlm
    
    # Run detection without translation (we'll do that after user selects)
    # Using a placeholder language since we won't use the translation yet
    result = run_snap_learn_vlm(image_bytes, "en", max_objects=max_objects)
    
    print(f"\n  ✅ Detection complete!")
    print(f"  📊 Image size: {result.image_width} × {result.image_height}")
    print(f"  🎯 Objects found: {result.total_objects}")
    
    return result


def display_objects(result) -> dict:
    """Display detected objects with selection numbers."""
    print_section("📋 STEP 1D: Choose Detected Object")
    
    if result.total_objects == 0:
        print("  ⚠️  No objects detected in this image.")
        print("  💡 Try a different image with clearer objects.")
        sys.exit(0)
    
    print(f"\n  {_ui('Select an object to translate:')}\n")
    
    objects_map = {}
    for i, obj in enumerate(result.objects, 1):
        objects_map[i] = obj
        display_label = _native_label_for_object(obj.canonical_tag)
        conf_bar = "█" * int(obj.confidence * 10) + "░" * (10 - int(obj.confidence * 10))
        print(f"    [{i}] {display_label:<20} {conf_bar} {obj.confidence:.0%}")
    
    print(f"\n    [0] Exit")
    
    return objects_map


def select_object(objects_map: dict):
    """Prompt user to select an object."""
    print_section("👆 Object Selection")
    
    while True:
        try:
            choice = input(f"\n  {_ui('Enter object number:')} ").strip()
            if choice == "0":
                print("\n  👋 Goodbye!")
                sys.exit(0)
            
            num = int(choice)
            if num in objects_map:
                selected = objects_map[num]
                print(f"\n  ✅ {_ui('Detected object')}: {_native_label_for_object(selected.canonical_tag)}")
                return selected
            else:
                print(f"  ❌ Invalid choice. Enter 1-{len(objects_map)} or 0 to exit.")
        except ValueError:
            print(f"  ❌ {_ui('Please enter a number.')}")


def display_languages(title: str, back_label: Optional[str] = None) -> dict:
    """Display available languages with selection numbers."""
    print_section(title)
    
    print(f"\n  {_ui('Available languages:')}\n")
    
    lang_map = {}
    for i, (code, name) in enumerate(SUPPORTED_LANGUAGES.items(), 1):
        lang_map[i] = code
        print(f"    [{i:2}] {name}")
    
    if back_label is not None:
        print(f"\n    [ 0] {back_label}")
    
    return lang_map


def select_language(lang_map: dict, allow_back: bool = True) -> Optional[str]:
    """Prompt user to select a language."""
    while True:
        try:
            choice = input(f"\n  {_ui('Enter language number:')} ").strip()
            if choice == "0" and allow_back:
                return None  # Go back
            
            num = int(choice)
            if num in lang_map:
                code = lang_map[num]
                name = SUPPORTED_LANGUAGES[code]
                print(f"\n  ✅ Selected: {name}")
                return code
            else:
                if allow_back:
                    print(f"  ❌ Invalid choice. Enter 1-{len(lang_map)} or 0 to go back.")
                else:
                    print(f"  ❌ Invalid choice. Enter 1-{len(lang_map)}.")
        except ValueError:
            print(f"  ❌ {_ui('Please enter a number.')}")


def select_session_languages() -> tuple[str, str]:
    """Prompt for native and initial target languages immediately after detection."""
    native_map = display_languages("🧭 STEP 1C: Select Native Language", back_label=None)
    native_lang = select_language(native_map, allow_back=False)
    _set_session_native_language(native_lang)

    target_map = display_languages("🌍 STEP 1E: Select Initial Target Language", back_label=None)
    while True:
        target_lang = select_language(target_map, allow_back=False)
        if target_lang != native_lang:
            return native_lang, target_lang
        print("  ❌ Native and target language cannot be the same for learning mode.")


def _translate_english_text_to_language(text: str, target_lang: str) -> str:
    """Translate English text to the target language for learner-facing hints."""
    if not text.strip() or target_lang == "en":
        return text

    target_code = TRANSLATOR_LANG_MAP.get(target_lang, target_lang)

    try:
        from deep_translator import GoogleTranslator

        translated = GoogleTranslator(source="en", target=target_code).translate(text)
        if translated and translated.strip():
            return translated.strip()
    except Exception:
        pass

    try:
        from deep_translator import MyMemoryTranslator

        translated = MyMemoryTranslator(source="en", target=target_code).translate(text)
        if translated and translated.strip():
            return translated.strip()
    except Exception:
        pass

    return text


def translate_label_to_native(source_word: str, native_lang: str) -> str:
    """Translate detected English label to the user's native language."""
    if native_lang == "en":
        return source_word

    try:
        from app.schemas.translation import FlashcardRequest
        from app.services.translation.service import build_flashcard

        flashcard = build_flashcard(
            FlashcardRequest(
                user_id="interactive-user",
                object_id="native_label",
                source_word=source_word,
                source_lang="en",
                target_lang=native_lang,
                proficiency_level="A2",
            ),
            interactive=False,
            raise_on_error=False,
        )
        if flashcard.translated_word.strip():
            return flashcard.translated_word.strip()
    except Exception:
        pass

    return _translate_english_text_to_language(source_word, native_lang)


def translate_object(source_word: str, target_lang: str):
    """
    Translate the selected object with human verification.
    
    Uses interactive mode - prompts user to verify/correct translations
    before saving to database.
    
    Returns:
        flashcard: The translation result, or None if user wants to skip
    """
    print_section("📚 STEP 3: Flashcard Generation")
    
    lang_name = SUPPORTED_LANGUAGES.get(target_lang, target_lang)
    print(f"\n  {_ui('Target word')}: '{source_word}' -> {lang_name}")
    
    from app.schemas.translation import FlashcardRequest
    from app.services.translation.service import build_flashcard, TranslationError
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            flashcard = build_flashcard(
                FlashcardRequest(
                    user_id="interactive-user",
                    object_id="obj_selected",
                    source_word=source_word,
                    source_lang="en",
                    target_lang=target_lang,
                    proficiency_level="A2",
                ),
                interactive=True,  # Enable human verification for new translations
                raise_on_error=True
            )
            
            source_labels = {
                "translation_memory": "📚 Database (verified)",
                "human_verified": "✅ Human Verified",
                "google_cloud": "☁️ Google Cloud Translation",
                "deepl": "🌐 DeepL",
                "google": "🔍 Google",
                "mymemory": "🧠 MyMemory fallback",
                "ollama": "🤖 Ollama fallback",
                "error_fallback": "⚠️ Error fallback",
            }
            source_label = source_labels.get(flashcard.translation_source, flashcard.translation_source)
            print(f"  ✅ {_ui('Flashcard generation complete.') if _SESSION_NATIVE_LANG != 'en' else 'Translation complete!'} (Source: {source_label})")
            return flashcard
            
        except TranslationError as e:
            print(f"\n  ❌ {_ui('Flashcard generation failed.')} {e}")
            
            if attempt < max_retries - 1:
                retry = input(f"\n  🔄 Retry? ({attempt + 1}/{max_retries}) [Y/n/skip]: ").strip().lower()
                if retry == 'n':
                    print("  ⏭️  Skipping translation...")
                    return None
                elif retry == 'skip':
                    return None
                print(f"  ⏳ Retrying ({attempt + 2}/{max_retries})...")
            else:
                print(f"\n  ⚠️  Max retries reached. Translation unavailable.")
                skip = input("  Skip this object? [Y/n]: ").strip().lower()
                if skip != 'n':
                    return None
                # Use fallback
                flashcard = build_flashcard(
                    FlashcardRequest(
                        user_id="interactive-user",
                        object_id="obj_selected",
                        source_word=source_word,
                        source_lang="en",
                        target_lang=target_lang,
                        proficiency_level="A2",
                    ),
                    interactive=False,  # No verification for fallback
                    raise_on_error=False
                )
                print(f"  ⚠️  Using fallback translation")
                return flashcard
    
    return None


def generate_audio(text: str, lang_code: str):
    """Generate TTS audio for the translated word."""
    print_section("🔊 STEP 4: Text-to-Speech")
    
    print(f"\n  Generating speech for target word '{text}'...")
    
    from app.schemas.speech import TTSRequest
    from app.services.tts.service import synthesize
    
    tts_result = synthesize(TTSRequest(
        text=text,
        lang_code=lang_code,
        voice="default",
        speed=1.0,
    ))

    global _SESSION_LAST_TTS_AUDIO_PATH
    _SESSION_LAST_TTS_AUDIO_PATH = str((_audio_dir() / f"{tts_result.audio_id}.wav").resolve())
    _prune_audio_storage([_SESSION_LAST_TTS_AUDIO_PATH, _SESSION_LAST_PRON_AUDIO_PATH or ""])
    
    engine_name = {"edge": "☁️ Edge-TTS (cloud)", "silence": "🔇 Fallback"}
    print(f"  ✅ Audio generated! (Engine: {engine_name.get(tts_result.engine, tts_result.engine)})")
    
    return tts_result


def display_flashcard(obj, flashcard, tts_result, target_lang: str):
    raise RuntimeError("Use display_flashcard_enriched instead")


def _tokenize_ipa_text(ipa_text: str) -> list[str]:
    cleaned = ipa_text.strip().strip("/")
    if not cleaned:
        return []

    try:
        from app.services.pronunciation import pronunciation_lab as pron_lab

        tokens = pron_lab._tokenize_ipa(cleaned.replace(".", " "))
        if tokens:
            return tokens
    except Exception:
        pass

    fallback = re.findall(r"[A-Za-z\u0250-\u02AF\u0300-\u036fːˤʰʲ̃͡]+", cleaned)
    return [token for token in fallback if token]


def _ipa_looks_weak(ipa: str, translated_word: str) -> bool:
    candidate = ipa.strip().lower()
    if not candidate or candidate in {"n/a", "na", "none"}:
        return True

    norm_ipa = re.sub(r"[^a-z0-9]", "", candidate)
    norm_word = re.sub(r"[^a-z0-9]", "", translated_word.lower())
    if norm_ipa and norm_ipa == norm_word:
        return True

    if len(candidate) > 64:
        return True

    return False


def _extract_pronunciation_module_phonemes(word: str, lang_code: str) -> tuple[list[str], str]:
    try:
        from app.services.pronunciation import pronunciation_lab as pron_lab

        return pron_lab._auto_extract_reference_phonemes(word, lang_code)
    except Exception as exc:
        return [], f"pronunciation module unavailable ({type(exc).__name__})"


def choose_best_phonetic_representation(
    translated_word: str,
    target_lang: str,
    flashcard_ipa: str,
) -> tuple[str, list[str], str]:
    """Choose the clearest phonetic guide between flashcard IPA and pronunciation module."""
    flash_tokens = _tokenize_ipa_text(flashcard_ipa)
    flash_reliable = bool(flash_tokens) and not _ipa_looks_weak(flashcard_ipa, translated_word)

    pron_tokens, pron_note = _extract_pronunciation_module_phonemes(translated_word, target_lang)
    note_low = pron_note.lower()
    pron_reliable = bool(pron_tokens) and "fallback" not in note_low and "failed" not in note_low

    if pron_reliable:
        return " ".join(pron_tokens), pron_tokens, f"Pronunciation module ({pron_note})"

    if flash_reliable:
        return flashcard_ipa.strip(), flash_tokens, "Flashcard IPA"

    if pron_tokens:
        return " ".join(pron_tokens), pron_tokens, f"Pronunciation module ({pron_note})"

    if flash_tokens:
        return flashcard_ipa.strip(), flash_tokens, "Flashcard IPA"

    literal = translated_word.strip() or "n/a"
    return literal, [literal], "Literal fallback"


def auto_play_audio(tts_result) -> None:
    """Auto-play generated TTS audio when possible."""
    audio_path = f"./data/audio/{tts_result.audio_id}.wav"
    abs_audio_path = os.path.abspath(audio_path)
    if not os.path.exists(abs_audio_path):
        print(f"  ⚠️  Audio file not found for auto-play: {abs_audio_path}")
        return

    print(f"  💡 Audio saved to: {abs_audio_path}")
    if os.name == "nt":
        try:
            os.startfile(abs_audio_path)
            print("  ▶️  Auto-playing target word audio...")
            return
        except Exception:
            pass

    print("  ℹ️  Auto-play is only automatic on Windows shell. Open the file above to play.")


def display_flashcard_enriched(
    obj,
    flashcard,
    tts_result,
    target_lang: str,
    native_lang: str,
    native_object_label: str,
    selected_phonetics: str,
    phonetic_source: str,
    example_translation_native: str,
) -> None:
    """Display final learning card with target and native-language context."""
    print_section("📚 FLASHCARD RESULT")

    target_name = _language_name(target_lang)
    native_name = _language_name(native_lang)

    print(f"\n  🖼️  {_ui('Detected object')} (English): {obj.canonical_tag}")
    print(f"  🏷️  {_ui('Detected object')} ({native_name}): {native_object_label}")
    print(f"\n  🌍 {_ui('Target language')}: {target_name}")
    print(f"  🔤 {_ui('Target word')}: {flashcard.translated_word}")
    print(f"  🔊 {_ui('Phonetic guide')}: {selected_phonetics}")
    print(f"  ℹ️  {_ui('Phonetic source')}: {phonetic_source}")
    print(f"\n  📝 Example ({target_name}): {flashcard.example_sentence}")
    print(f"  📝 Translation ({native_name}): {example_translation_native}")
    print(f"\n  🎧 TTS file: {tts_result.audio_id}.wav")
    print(f"  ⏱️  TTS duration: {tts_result.duration_ms} ms")


def _ensure_pronunciation_runtime() -> None:
    """Prepare pronunciation runtime lazily for opt-in scoring."""
    global _PRONUNCIATION_READY
    if _PRONUNCIATION_READY:
        return

    from app.services.pronunciation.service import warmup_local_pronunciation_models
    from app.services.pronunciation import pronunciation_lab as pron_lab

    print_section("🗣️ STEP 5A: Preparing Pronunciation Runtime")
    runtime_message = pron_lab._configure_runtime_gpu_preferred()
    print(f"  {runtime_message}")

    warmup_ok, warmup_msg = warmup_local_pronunciation_models()
    if warmup_ok:
        print(f"  ✅ {warmup_msg}")
    else:
        print(f"  ⚠️ {warmup_msg}")

    _PRONUNCIATION_READY = True


def run_pronunciation_assessment(
    target_word: str,
    target_lang: str,
    reference_phonemes: list[str],
) -> bool:
    """Run pronunciation assessment loop until green or user exits."""
    _ensure_pronunciation_runtime()

    from app.services.pronunciation import pronunciation_lab as pron_lab

    print_section("🗣️ STEP 5B: Pronunciation Assessment")
    print(f"\n  {_ui('Target word')}: {target_word}")
    print(f"  Reference phonemes: {' '.join(reference_phonemes)}")

    sample_rate = 16000
    attempt = 1
    previous_response = None

    while True:
        print_section(f"Attempt {attempt}")
        try:
            audio = pron_lab._record_push_to_talk(sample_rate=sample_rate, channels=1)
        except Exception as exc:
            print(f"  ❌ Recording failed: {type(exc).__name__}")
            if _prompt_yes_no("  Retry recording?", default=True):
                continue
            return False

        audio_path = pron_lab._save_attempt_audio(audio, sample_rate=sample_rate, attempt=attempt)
        global _SESSION_LAST_PRON_AUDIO_PATH
        _SESSION_LAST_PRON_AUDIO_PATH = str(Path(audio_path).resolve())
        _prune_audio_storage([_SESSION_LAST_TTS_AUDIO_PATH or "", _SESSION_LAST_PRON_AUDIO_PATH])
        print(f"  💾 Saved attempt audio: {audio_path}")

        if _prompt_yes_no("  Replay this recording?", default=False):
            pron_lab._replay_audio(audio, sample_rate=sample_rate)

        wav_bytes = pron_lab._audio_to_wav_bytes(audio, sample_rate=sample_rate)
        response = pron_lab._score_attempt(
            lang_code=target_lang,
            target_word=target_word,
            reference_phonemes=reference_phonemes,
            wav_bytes=wav_bytes,
        )

        pron_lab._print_attempt_report(response)
        pron_lab._print_alignment_notes(response)

        if previous_response is not None:
            pron_lab._print_improvement(previous_response, response)

        if response.overall_level == "green":
            print("\n  ✅ Great work. Pronunciation target achieved.")
            return True

        pron_lab._print_correction_advice(response, target_lang)
        previous_response = response

        if not _prompt_yes_no("\n  Try pronunciation again?", default=True):
            return False

        attempt += 1


def run_target_language_cycle(selected_obj, native_lang: str, target_lang: str, native_object_label: str) -> bool:
    """Run flashcard -> TTS -> optional pronunciation for one target language."""
    flashcard = translate_object(selected_obj.canonical_tag, target_lang)
    if flashcard is None:
        print("\n  ⏭️  Skipping this language due to translation failure.")
        return False

    selected_phonetics, reference_phonemes, phonetic_source = choose_best_phonetic_representation(
        translated_word=flashcard.translated_word,
        target_lang=target_lang,
        flashcard_ipa=flashcard.ipa,
    )

    # Flashcard example_translation is English. Convert it to user's native language.
    example_translation_native = _translate_english_text_to_language(
        flashcard.example_translation,
        native_lang,
    )

    tts_result = generate_audio(flashcard.translated_word, target_lang)
    auto_play_audio(tts_result)

    display_flashcard_enriched(
        obj=selected_obj,
        flashcard=flashcard,
        tts_result=tts_result,
        target_lang=target_lang,
        native_lang=native_lang,
        native_object_label=native_object_label,
        selected_phonetics=selected_phonetics,
        phonetic_source=phonetic_source,
        example_translation_native=example_translation_native,
    )

    if _prompt_yes_no("\nDo you want to test your pronunciation for this word?", default=False):
        run_pronunciation_assessment(
            target_word=flashcard.translated_word,
            target_lang=target_lang,
            reference_phonemes=reference_phonemes,
        )

    return True


def save_artifacts_info():
    """Display info about saved artifacts."""
    print_section("💾 SAVED ARTIFACTS")
    
    artifacts_dir = "./data/artifacts"
    audio_dir = "./data/audio"
    
    print(f"\n  📁 Detection results: {os.path.abspath(artifacts_dir)}/")
    if os.path.exists(artifacts_dir):
        files = os.listdir(artifacts_dir)
        for f in sorted(files)[:5]:
            print(f"     • {f}")
        if len(files) > 5:
            print(f"     ... and {len(files) - 5} more files")
    
    print(f"\n  📁 {_ui('Audio files')}: {os.path.abspath(audio_dir)}/")


def main():
    """Main interactive pipeline."""
    parser = argparse.ArgumentParser(description="SnapTalk Interactive Pipeline")
    parser.add_argument("--image", help="Path to image file")
    args = parser.parse_args()
    
    # Default test images
    default_images = ["test1.jpeg", "test2.jpg", "test3.jpg", "test4.jpg"]
    
    clear_screen()
    print_header()
    _start_audio_session()
    
    # Get image path
    if args.image:
        image_path = args.image
    else:
        print("\n  No image specified. Available test images:")
        for img in default_images:
            if os.path.exists(img):
                print(f"    • {img}")
        image_path = input("\n  Enter image path (or press Enter for test1.jpeg): ").strip()
        if not image_path:
            image_path = "test1.jpeg"
    
    # Run pipeline
    image_bytes = load_image(image_path)
    result = detect_objects(image_bytes)

    # Prompt native + target language immediately after detection.
    native_lang, initial_target_lang = select_session_languages()
    
    # Interactive selection loop
    while True:
        objects_map = display_objects(result)
        selected_obj = select_object(objects_map)

        print_section("🏷️ STEP 2: Native Language Detection Label")
        native_object_label = translate_label_to_native(selected_obj.canonical_tag, native_lang)
        print(f"\n  {_ui('Detected object')} (English): {selected_obj.canonical_tag}")
        print(f"  {_ui('Detected object')} ({_language_name(native_lang)}): {native_object_label}")

        current_target = initial_target_lang
        while True:
            run_target_language_cycle(
                selected_obj=selected_obj,
                native_lang=native_lang,
                target_lang=current_target,
                native_object_label=native_object_label,
            )

            # Crucial multi-language loop for the same selected object.
            if not _prompt_yes_no("\nDo you want to translate this word into another language?", default=False):
                break

            lang_map = display_languages(
                title="🌐 STEP 6: Select Another Target Language",
                back_label="Stop additional language loop",
            )
            next_target = select_language(lang_map, allow_back=True)
            if next_target is None:
                break
            if next_target == current_target:
                print("  ℹ️ This language is already active. Choose a different target language.")
                continue
            if next_target == native_lang:
                print("  ❌ Target language cannot match native language in learning mode.")
                continue

            current_target = next_target

        save_artifacts_info()

        if not _prompt_yes_no("\nDo you want to select another detected object from this image?", default=True):
            print(f"\n  👋 {_ui('Thanks for using SnapTalk! Goodbye!')}\n")
            return


if __name__ == "__main__":
    main()
