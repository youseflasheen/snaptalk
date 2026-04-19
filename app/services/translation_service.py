import difflib
import html
import json
import os
import re
import sqlite3
import threading
import uuid
from typing import Optional, Tuple

import httpx

from app.core.config import settings
from app.schemas.translation import FlashcardRequest, FlashcardResponse


class TranslationError(Exception):
    """Raised when all translation engines fail."""


_vocab_terms: Optional[set[str]] = None
_vocab_lock = threading.Lock()

_MANUAL_SOURCE_CORRECTIONS = {
    "hoddie": "hoodie",
    "hoody": "hoodie",
}

_EMERGENCY_TRANSLATION_MAP = {
    ("hoodie", "ar"): "هودي",
    ("hoodie", "ru"): "худи",
    ("hoodie", "es"): "sudadera con capucha",
    ("hoodie", "fr"): "sweat à capuche",
    ("hoodie", "de"): "kapuzenpullover",
    ("computer", "ar"): "حاسوب",
    ("computer", "ru"): "компьютер",
    ("pen", "ar"): "قلم",
    ("ball", "ar"): "كرة",
}


def _language_prompt_context(target_lang: str) -> tuple[str, str]:
    lang_info = {
        "es": ("Spanish", "Use Spanish script."),
        "fr": ("French", "Use French script with accents when needed."),
        "ja": ("Japanese", "Use Japanese script only."),
        "de": ("German", "Use German script."),
        "it": ("Italian", "Use Italian script."),
        "pt": ("Portuguese", "Use Portuguese script."),
        "ar": ("Arabic", "Use Arabic script only."),
        "zh": ("Chinese", "Use Simplified Chinese script only."),
        "ko": ("Korean", "Use Hangul script only."),
        "ru": ("Russian", "Use Cyrillic script only."),
        "nl": ("Dutch", "Use Dutch script."),
        "tr": ("Turkish", "Use Turkish script."),
    }
    return lang_info.get(target_lang, (target_lang, "Use native script."))


def _normalized_token(text: str) -> str:
    return "".join(ch.lower() for ch in text if ch.isalnum())


def _load_vocab_terms() -> set[str]:
    global _vocab_terms

    if _vocab_terms is not None:
        return _vocab_terms

    with _vocab_lock:
        if _vocab_terms is not None:
            return _vocab_terms

        terms: set[str] = {
            "hoodie",
            "sweatshirt",
            "computer",
            "laptop",
            "phone",
            "bottle",
            "table",
            "chair",
            "pen",
            "book",
            "ball",
        }

        candidate_paths = [settings.yolo_world_vocab_path, "./data/yolo_world_vocab.txt"]
        for path in candidate_paths:
            if not path or not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    for line in handle:
                        raw = line.strip().lower()
                        if not raw or raw.startswith("#"):
                            continue
                        tokens = re.findall(r"[a-z]+", raw)
                        terms.update(token for token in tokens if len(token) >= 3)
                break
            except Exception:
                continue

        _vocab_terms = terms
        return terms


def _normalize_source_word(source_word: str) -> str:
    normalized = " ".join(source_word.strip().lower().split())
    if not normalized:
        return source_word.strip()

    vocab = _load_vocab_terms()
    vocab_list = sorted(vocab)

    out_tokens: list[str] = []
    for token in normalized.split():
        clean = re.sub(r"[^a-z]", "", token)
        if not clean:
            continue

        if clean in _MANUAL_SOURCE_CORRECTIONS:
            out_tokens.append(_MANUAL_SOURCE_CORRECTIONS[clean])
            continue

        if clean in vocab:
            out_tokens.append(clean)
            continue

        matches = difflib.get_close_matches(clean, vocab_list, n=1, cutoff=0.84)
        out_tokens.append(matches[0] if matches else clean)

    return " ".join(out_tokens) if out_tokens else normalized


def _source_word_candidates(source_word: str) -> list[str]:
    original = " ".join(source_word.strip().lower().split())
    corrected = _normalize_source_word(original)

    candidates: list[str] = []
    # Prefer corrected form first so typo normalization is used before misspelled variants.
    for candidate in [corrected, original]:
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    return candidates or [source_word.strip()]


def _contains_source_script(text: str, target_lang: str) -> bool:
    script_patterns = {
        "ar": r"[\u0600-\u06FF]",
        "ru": r"[\u0400-\u04FF]",
        "zh": r"[\u4E00-\u9FFF]",
        "ja": r"[\u3040-\u30FF\u4E00-\u9FFF]",
        "ko": r"[\uAC00-\uD7AF]",
    }
    pattern = script_patterns.get(target_lang)
    if not pattern:
        return False
    return bool(re.search(pattern, text))


def _translation_looks_low_quality(source_word: str, translated_word: str, target_lang: str) -> bool:
    if not translated_word:
        return True

    cleaned = translated_word.strip()
    if not cleaned:
        return True

    low = cleaned.lower()
    blocked_phrases = (
        "i can't",
        "i cannot",
        "cannot provide",
        "can't provide",
        "illegal",
        "harmful",
        "child sexual",
        "anything else i can help",
        "sorry",
    )
    if any(phrase in low for phrase in blocked_phrases):
        return True

    if low.endswith(f"_{target_lang.lower()}"):
        return True
    if re.fullmatch(r"[a-z0-9_-]+_[a-z]{2,5}", low):
        return True

    normalized_source = _normalized_token(source_word)
    normalized_target = _normalized_token(cleaned)
    if not normalized_target:
        return True
    if normalized_source and normalized_source == normalized_target:
        return True

    if target_lang in {"ar", "ru", "zh", "ja", "ko"} and not _contains_source_script(cleaned, target_lang):
        return True

    return False


def _coerce_known_term_translation(source_word: str, target_lang: str, translated_word: str) -> tuple[str, bool]:
    """Prefer trusted vocabulary mappings when model output is clearly off for known terms."""
    normalized_source = _normalize_source_word(source_word)
    expected = _EMERGENCY_TRANSLATION_MAP.get((normalized_source, target_lang))
    if not expected:
        return translated_word, False

    candidate_norm = _normalized_token(translated_word)
    expected_norm = _normalized_token(expected)
    if not candidate_norm:
        return expected, True

    if expected_norm in candidate_norm or candidate_norm in expected_norm:
        return translated_word, False

    similarity = difflib.SequenceMatcher(None, candidate_norm, expected_norm).ratio()
    if similarity >= 0.55:
        return translated_word, False

    return expected, True


def _ipa_looks_low_quality(ipa: str, translated_word: str, target_lang: str) -> bool:
    if not ipa:
        return True

    cleaned_ipa = ipa.strip()
    low = cleaned_ipa.lower()

    if low in {"n/a", "na", "none"}:
        return True

    blocked_phrases = (
        "i can't",
        "i cannot",
        "cannot provide",
        "can't provide",
        "illegal",
        "harmful",
        "child sexual",
        "anything else i can help",
        "sorry",
    )
    if any(phrase in low for phrase in blocked_phrases):
        return True

    if len(cleaned_ipa) > 64 or len(cleaned_ipa.split()) > 8:
        return True

    if _normalized_token(cleaned_ipa) == _normalized_token(translated_word):
        return True

    if _contains_source_script(cleaned_ipa, target_lang):
        return True

    return False


def _refine_ipa_with_ollama(translated_word: str, target_lang: str) -> Optional[str]:
    endpoint = f"{settings.llm_base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.llm_api_key}",
        "Content-Type": "application/json",
    }

    target_name, script_hint = _language_prompt_context(target_lang)
    prompt = (
        f"Provide the IPA pronunciation for the {target_name} word '{translated_word}'. "
        f"{script_hint} IPA must not be in source script. "
        "Return only valid JSON with one key: ipa."
    )
    body = {
        "model": settings.translation_default_model,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        with httpx.Client(timeout=20.0) as client:
            response = client.post(endpoint, headers=headers, json=body)
            response.raise_for_status()
            payload = response.json()

        content = payload["choices"][0]["message"]["content"]
        parsed = _extract_json_object(content)
        if parsed and parsed.get("ipa"):
            candidate = str(parsed.get("ipa", "")).strip().strip("/")
            if candidate and not _ipa_looks_low_quality(candidate, translated_word, target_lang):
                return candidate

        candidate = str(content).strip().strip("/")
        if candidate and not _ipa_looks_low_quality(candidate, translated_word, target_lang):
            return candidate
    except Exception:
        return None

    return None


def _resolve_ipa(translated_word: str, target_lang: str, candidate_ipa: str, allow_llm: bool = True) -> str:
    if candidate_ipa and not _ipa_looks_low_quality(candidate_ipa, translated_word, target_lang):
        return candidate_ipa

    if allow_llm:
        refined = _refine_ipa_with_ollama(translated_word, target_lang)
        if refined:
            return refined

    fallback = _generate_fallback_ipa(translated_word, target_lang)
    return fallback or "n/a"


def _generate_examples_with_ollama(
    source_word: str,
    translated_word: str,
    target_lang: str,
    proficiency_level: str,
) -> Optional[tuple[str, str]]:
    endpoint = f"{settings.llm_base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.llm_api_key}",
        "Content-Type": "application/json",
    }
    target_name, script_hint = _language_prompt_context(target_lang)
    prompt = (
        f"Create one short {target_name} sentence using the word '{translated_word}'. "
        f"Learner level: {proficiency_level}. {script_hint} "
        "Return only valid JSON with keys: example_sentence, example_translation. "
        "example_sentence must be in target language, example_translation must be English."
    )
    body = {
        "model": settings.translation_default_model,
        "temperature": 0.2,
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        with httpx.Client(timeout=20.0) as client:
            response = client.post(endpoint, headers=headers, json=body)
            response.raise_for_status()
            payload = response.json()

        content = payload["choices"][0]["message"]["content"]
        parsed = _extract_json_object(content)
        if parsed:
            sentence = str(parsed.get("example_sentence", "")).strip()
            meaning = str(parsed.get("example_translation", "")).strip()
            if sentence and meaning:
                if sentence.lower().startswith("this is "):
                    return None
                if target_lang in {"ar", "ru", "zh", "ja", "ko"} and not _contains_source_script(sentence, target_lang):
                    return None
                return sentence, meaning
    except Exception:
        return None

    return None


def _template_example_sentence(source_word: str, translated_word: str, target_lang: str) -> tuple[str, str]:
    templates = {
        "es": (f"Veo {translated_word}.", f"I see {source_word}."),
        "fr": (f"Je vois {translated_word}.", f"I see {source_word}."),
        "de": (f"Ich sehe {translated_word}.", f"I see {source_word}."),
        "it": (f"Vedo {translated_word}.", f"I see {source_word}."),
        "pt": (f"Eu vejo {translated_word}.", f"I see {source_word}."),
        "ar": (f"أرى {translated_word}.", f"I see {source_word}."),
        "ru": (f"Я вижу {translated_word}.", f"I see {source_word}."),
        "nl": (f"Ik zie {translated_word}.", f"I see {source_word}."),
        "tr": (f"{translated_word} görüyorum.", f"I see {source_word}."),
        "ja": (f"{translated_word}を見ます。", f"I see {source_word}."),
        "zh": (f"我看到{translated_word}。", f"I see {source_word}."),
        "ko": (f"{translated_word}를 봐요.", f"I see {source_word}."),
    }
    return templates.get(target_lang, (f"{translated_word}.", f"It means '{source_word}'."))


def _ensure_translation_db() -> None:
    db_path = settings.translation_db_path
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS translation_memory (
                source_word TEXT NOT NULL,
                source_lang TEXT NOT NULL,
                target_lang TEXT NOT NULL,
                translated_word TEXT NOT NULL,
                ipa TEXT NOT NULL,
                PRIMARY KEY (source_word, source_lang, target_lang)
            )
            """
        )
        conn.commit()


def _seed_if_empty() -> None:
    """Seed translation memory from JSON file if empty."""
    with sqlite3.connect(settings.translation_db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM translation_memory").fetchone()[0]
        if count > 0:
            return

    seed_path = "./data/seed_translations.json"
    if not os.path.exists(seed_path):
        _seed_basic_fallback()
        return

    try:
        with open(seed_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        all_langs = ["es", "fr", "ja", "de", "it", "pt", "ar", "zh", "ko", "ru", "nl", "tr"]
        seeds: list[tuple[str, str, str, str, str]] = []

        for item in data.get("translations", []):
            en_word = str(item.get("en", "")).strip().lower()
            if not en_word:
                continue

            for lang in all_langs:
                if lang not in item:
                    continue
                val = item[lang]
                if isinstance(val, list) and len(val) >= 2:
                    translated, ipa = str(val[0]), str(val[1])
                else:
                    translated = str(val)
                    ipa = _generate_fallback_ipa(translated, lang)
                seeds.append((en_word, "en", lang, translated, ipa))

        with sqlite3.connect(settings.translation_db_path) as conn:
            conn.executemany(
                """
                INSERT OR IGNORE INTO translation_memory
                (source_word, source_lang, target_lang, translated_word, ipa)
                VALUES (?, ?, ?, ?, ?)
                """,
                seeds,
            )
            conn.commit()
    except Exception as exc:
        print(f"[Translation] Warning: could not seed from JSON: {exc}")
        _seed_basic_fallback()


def _seed_basic_fallback() -> None:
    seeds = [
        ("apple", "en", "es", "manzana", "man.sa.na"),
        ("book", "en", "es", "libro", "li.bro"),
        ("apple", "en", "fr", "pomme", "pom"),
        ("book", "en", "fr", "livre", "livr"),
        ("apple", "en", "ja", "りんご", "ri.n.go"),
        ("book", "en", "ja", "本", "hon"),
    ]

    with sqlite3.connect(settings.translation_db_path) as conn:
        conn.executemany(
            """
            INSERT OR IGNORE INTO translation_memory
            (source_word, source_lang, target_lang, translated_word, ipa)
            VALUES (?, ?, ?, ?, ?)
            """,
            seeds,
        )
        conn.commit()


def _lookup_translation(source_word: str, source_lang: str, target_lang: str) -> Optional[Tuple[str, str]]:
    """Lookup translation from cache and repair missing IPA on read."""
    with sqlite3.connect(settings.translation_db_path) as conn:
        row = conn.execute(
            """
            SELECT translated_word, ipa
            FROM translation_memory
            WHERE source_word = ? AND source_lang = ? AND target_lang = ?
            """,
            (source_word.lower(), source_lang.lower(), target_lang.lower()),
        ).fetchone()

    if row is None:
        return None

    translated_word, ipa = str(row[0]), str(row[1])
    resolved_ipa = _resolve_ipa(translated_word, target_lang, ipa, allow_llm=False)

    if resolved_ipa != ipa:
        try:
            with sqlite3.connect(settings.translation_db_path) as conn:
                conn.execute(
                    """
                    UPDATE translation_memory
                    SET ipa = ?
                    WHERE source_word = ? AND source_lang = ? AND target_lang = ?
                    """,
                    (resolved_ipa, source_word.lower(), source_lang.lower(), target_lang.lower()),
                )
                conn.commit()
        except Exception:
            pass

    return translated_word, resolved_ipa


def _generate_fallback_ipa(word: str, target_lang: str) -> str:
    """Generate lightweight fallback pronunciation when true IPA is unavailable."""
    if not word:
        return "n/a"

    cleaned = "".join(ch for ch in word.strip() if ch not in " -'\"")
    if not cleaned:
        return "n/a"

    if target_lang == "ar":
        arabic_sounds = {
            "ا": "aa",
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
            "ع": "a",
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
        }
        phones = [arabic_sounds.get(ch, "") for ch in cleaned]
        phones = [p for p in phones if p]
        return ".".join(phones) if phones else "n/a"

    if target_lang == "ru":
        cyrillic = {
            "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "ye", "ё": "yo", "ж": "zh",
            "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o",
            "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "kh", "ц": "ts",
            "ч": "ch", "ш": "sh", "щ": "shch", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
        }
        phones = [cyrillic.get(ch, ch.lower()) for ch in cleaned.lower()]
        phones = [p for p in phones if p]
        return ".".join(phones) if phones else "n/a"

    latin_heavy = {"es", "fr", "de", "it", "pt", "nl", "tr", "pl"}
    if target_lang in latin_heavy:
        mapped: list[str] = []
        text = cleaned.lower()
        i = 0
        while i < len(text):
            ch = text[i]
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if ch == "c" and nxt in {"e", "i"}:
                mapped.append("s")
            elif ch == "c":
                mapped.append("k")
            elif ch == "x":
                mapped.append("ks")
            elif ch == "j":
                mapped.append("h")
            elif ch.isalnum():
                mapped.append(ch)
            i += 1
        return ".".join(mapped) if mapped else "n/a"

    generic = [ch.lower() for ch in cleaned if ch.strip()]
    return ".".join(generic) if generic else "n/a"


def _cache_translation(source_word: str, source_lang: str, target_lang: str, translated_word: str, ipa: str) -> None:
    with sqlite3.connect(settings.translation_db_path) as conn:
        conn.execute(
            """
            INSERT INTO translation_memory (source_word, source_lang, target_lang, translated_word, ipa)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(source_word, source_lang, target_lang)
            DO UPDATE SET translated_word = excluded.translated_word, ipa = excluded.ipa
            """,
            (source_word.lower(), source_lang.lower(), target_lang.lower(), translated_word, ipa),
        )
        conn.commit()


def _build_example_sentence(
    source_word: str,
    translated_word: str,
    target_lang: str,
    proficiency_level: str = "A2",
    use_llm: bool = True,
) -> tuple[str, str]:
    if use_llm:
        generated = _generate_examples_with_ollama(
            source_word=source_word,
            translated_word=translated_word,
            target_lang=target_lang,
            proficiency_level=proficiency_level,
        )
        if generated is not None:
            return generated
    return _template_example_sentence(source_word, translated_word, target_lang)


def _google_cloud_translate(source_word: str, target_lang: str) -> Optional[Tuple[str, str]]:
    """Translate using official Google Cloud Translation API."""
    project_id = settings.google_cloud_project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        return None

    credentials_path = settings.google_cloud_credentials_path.strip()
    if credentials_path and os.path.exists(credentials_path):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

    lang_map = {
        "es": "es",
        "fr": "fr",
        "de": "de",
        "it": "it",
        "pt": "pt",
        "nl": "nl",
        "pl": "pl",
        "ru": "ru",
        "ja": "ja",
        "zh": "zh-CN",
        "ko": "ko",
        "ar": "ar",
        "tr": "tr",
    }
    target_code = lang_map.get(target_lang, target_lang)
    location = settings.google_cloud_location or "global"

    try:
        from google.cloud import translate

        client = translate.TranslationServiceClient()
        parent = f"projects/{project_id}/locations/{location}"
        response = client.translate_text(
            request={
                "parent": parent,
                "contents": [source_word],
                "mime_type": "text/plain",
                "source_language_code": "en",
                "target_language_code": target_code,
            },
            timeout=settings.google_cloud_timeout_seconds,
        )

        translations = response.translations or []
        if not translations:
            return None

        translated = html.unescape(str(translations[0].translated_text or "")).strip()
        if not _translation_looks_low_quality(source_word, translated, target_lang):
            return translated, "n/a"
    except ImportError:
        print("[Translation] google-cloud-translate is not installed")
    except Exception as exc:
        print(f"[Translation] Google Cloud failed: {type(exc).__name__}")

    return None


def _google_translate(source_word: str, target_lang: str) -> Optional[Tuple[str, str]]:
    """Translate using deep-translator Google backend (fallback after Google Cloud)."""
    try:
        from deep_translator import GoogleTranslator

        lang_map = {
            "es": "es",
            "fr": "fr",
            "de": "de",
            "it": "it",
            "pt": "pt",
            "nl": "nl",
            "pl": "pl",
            "ru": "ru",
            "ja": "ja",
            "zh": "zh-CN",
            "ko": "ko",
            "ar": "ar",
            "tr": "tr",
        }
        target_code = lang_map.get(target_lang, target_lang)

        translated = GoogleTranslator(source="en", target=target_code).translate(source_word)
        if translated and translated.strip():
            cleaned = translated.strip()
            if not _translation_looks_low_quality(source_word, cleaned, target_lang):
                return cleaned, "n/a"
    except ImportError:
        print("[Translation] deep-translator is not installed")
    except Exception as exc:
        print(f"[Translation] Google failed: {exc}")

    return None


def _deepl_translate(source_word: str, target_lang: str) -> Optional[Tuple[str, str]]:
    """Translate using DeepL API (second fallback stage)."""
    api_key = settings.deepl_api_key or os.getenv("DEEPL_API_KEY")
    if not api_key:
        return None

    deepl_lang_map = {
        "es": "ES",
        "fr": "FR",
        "de": "DE",
        "it": "IT",
        "pt": "PT-BR",
        "nl": "NL",
        "pl": "PL",
        "ru": "RU",
        "ja": "JA",
        "zh": "ZH",
        "ko": "KO",
        "tr": "TR",
    }
    target_code = deepl_lang_map.get(target_lang)
    if not target_code:
        return None

    url = "https://api-free.deepl.com/v2/translate"
    headers = {
        "Authorization": f"DeepL-Auth-Key {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"text": [source_word], "source_lang": "EN", "target_lang": target_code}

    try:
        with httpx.Client(timeout=15.0) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            body = response.json()

        translations = body.get("translations") or []
        if translations:
            translated = str(translations[0].get("text", "")).strip()
            if translated and not _translation_looks_low_quality(source_word, translated, target_lang):
                return translated, "n/a"
    except Exception as exc:
        print(f"[Translation] DeepL failed: {exc}")

    return None


def _mymemory_translate(source_word: str, target_lang: str) -> Optional[Tuple[str, str]]:
    """Translate using MyMemory as an additional fallback."""
    try:
        from deep_translator import MyMemoryTranslator

        mymemory_map = {
            "es": "es",
            "fr": "fr",
            "de": "de",
            "it": "it",
            "pt": "pt",
            "nl": "nl",
            "ru": "ru",
            "ja": "ja",
            "zh": "zh-CN",
            "ko": "ko",
            "ar": "ar",
            "tr": "tr",
        }
        target_code = mymemory_map.get(target_lang)
        if not target_code:
            return None

        translated = MyMemoryTranslator(source="en", target=target_code).translate(source_word)
        if translated and translated.strip():
            cleaned = translated.strip()
            if not _translation_looks_low_quality(source_word, cleaned, target_lang):
                return cleaned, "n/a"
    except Exception as exc:
        print(f"[Translation] MyMemory failed: {type(exc).__name__}")

    return None


def _ollama_translate(
    source_word: str,
    source_lang: str,
    target_lang: str,
    proficiency_level: str,
) -> Optional[Tuple[str, str, str, str]]:
    """Translate with local Ollama as the final fallback stage."""
    _ = source_lang
    endpoint = f"{settings.llm_base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.llm_api_key}",
        "Content-Type": "application/json",
    }

    target_name, script_hint = _language_prompt_context(target_lang)

    prompt = (
        f"Translate the English word '{source_word}' into {target_name}. "
        f"Proficiency level is {proficiency_level}. {script_hint} "
        "Return only valid JSON with keys: translated_word, ipa, example_sentence, example_translation."
    )

    body = {
        "model": settings.translation_default_model,
        "temperature": 0.1,
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        with httpx.Client(timeout=45.0) as client:
            response = client.post(endpoint, headers=headers, json=body)
            response.raise_for_status()
            payload = response.json()

        content = payload["choices"][0]["message"]["content"]
        parsed = _extract_json_object(content)
        if parsed and parsed.get("translated_word"):
            translated = str(parsed.get("translated_word", "")).strip()
            if translated and not _translation_looks_low_quality(source_word, translated, target_lang):
                return (
                    translated,
                    str(parsed.get("ipa", "n/a")).strip() or "n/a",
                    str(parsed.get("example_sentence", "")).strip(),
                    str(parsed.get("example_translation", "")).strip(),
                )
    except Exception as exc:
        print(f"[Translation] Ollama failed: {exc}")

    return None


def _extract_json_object(raw: str) -> Optional[dict]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        parsed = json.loads(text[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _fetch_translation(
    source_word: str,
    source_lang: str,
    target_lang: str,
    proficiency_level: str,
) -> tuple[str, str, str, str, str]:
    """
    Fallback order per source candidate:
    DeepL -> Google Cloud (official) -> Google (deep-translator) -> MyMemory -> Ollama.

    Returns:
        (translated_word, ipa, example_sentence, example_translation, source_engine)
    """
    google_mode = (settings.translation_google_mode or "official_with_fallback").strip().lower()
    use_google_cloud_primary = google_mode in {"", "official", "official_only", "official_with_fallback", "auto"}
    use_google_wrapper = google_mode in {"", "official", "official_with_fallback", "deep_translator_only", "auto"}
    if google_mode == "official_only":
        use_google_wrapper = False

    for candidate_source in _source_word_candidates(source_word):
        deepl_result = _deepl_translate(candidate_source, target_lang)
        if deepl_result:
            translated, _ = deepl_result
            translated, used_known_override = _coerce_known_term_translation(
                candidate_source,
                target_lang,
                translated,
            )
            ipa = _resolve_ipa(translated, target_lang, "n/a")
            ex_sentence, ex_translation = _build_example_sentence(
                candidate_source,
                translated,
                target_lang,
                proficiency_level=proficiency_level,
                use_llm=True,
            )
            source = "error_fallback" if used_known_override else "deepl"
            return translated, ipa, ex_sentence, ex_translation, source

        if use_google_cloud_primary:
            google_cloud_result = _google_cloud_translate(candidate_source, target_lang)
            if google_cloud_result:
                translated, _ = google_cloud_result
                translated, used_known_override = _coerce_known_term_translation(
                    candidate_source,
                    target_lang,
                    translated,
                )
                ipa = _resolve_ipa(translated, target_lang, "n/a")
                ex_sentence, ex_translation = _build_example_sentence(
                    candidate_source,
                    translated,
                    target_lang,
                    proficiency_level=proficiency_level,
                    use_llm=True,
                )
                source = "error_fallback" if used_known_override else "google_cloud"
                return translated, ipa, ex_sentence, ex_translation, source

        if use_google_wrapper:
            google_result = _google_translate(candidate_source, target_lang)
            if google_result:
                translated, _ = google_result
                translated, used_known_override = _coerce_known_term_translation(
                    candidate_source,
                    target_lang,
                    translated,
                )
                ipa = _resolve_ipa(translated, target_lang, "n/a")
                ex_sentence, ex_translation = _build_example_sentence(
                    candidate_source,
                    translated,
                    target_lang,
                    proficiency_level=proficiency_level,
                    use_llm=True,
                )
                source = "error_fallback" if used_known_override else "google"
                return translated, ipa, ex_sentence, ex_translation, source

        mymemory_result = _mymemory_translate(candidate_source, target_lang)
        if mymemory_result:
            translated, _ = mymemory_result
            translated, used_known_override = _coerce_known_term_translation(
                candidate_source,
                target_lang,
                translated,
            )
            ipa = _resolve_ipa(translated, target_lang, "n/a")
            ex_sentence, ex_translation = _build_example_sentence(
                candidate_source,
                translated,
                target_lang,
                proficiency_level=proficiency_level,
                use_llm=False,
            )
            source = "error_fallback" if used_known_override else "mymemory"
            return translated, ipa, ex_sentence, ex_translation, source

        ollama_result = _ollama_translate(candidate_source, source_lang, target_lang, proficiency_level)
        if ollama_result:
            translated, ipa, ex_sentence, ex_translation = ollama_result
            translated, used_known_override = _coerce_known_term_translation(
                candidate_source,
                target_lang,
                translated,
            )
            ipa = _resolve_ipa(translated, target_lang, ipa)
            if not ex_sentence or not ex_translation:
                ex_sentence, ex_translation = _build_example_sentence(
                    candidate_source,
                    translated,
                    target_lang,
                    proficiency_level=proficiency_level,
                    use_llm=False,
                )
            source = "error_fallback" if used_known_override else "ollama"
            return translated, ipa, ex_sentence, ex_translation, source

    raise TranslationError(f"All translation engines failed for '{source_word}' -> {target_lang}")


def build_flashcard(
    req: FlashcardRequest,
    interactive: bool = False,
    raise_on_error: bool = False,
) -> FlashcardResponse:
    """Build flashcard using cache first, then DeepL-first fallback chain."""
    _ensure_translation_db()
    _seed_if_empty()

    source_lang = req.source_lang.strip().lower()
    target_lang = req.target_lang.strip().lower()
    source_word = req.source_word.strip()

    if source_lang == target_lang:
        translated_word = source_word
        ipa = _resolve_ipa(translated_word, target_lang, "n/a", allow_llm=False)

        if target_lang == "en":
            example_sentence = f"I see {translated_word}."
            example_translation = example_sentence
        else:
            example_sentence, example_translation = _build_example_sentence(
                source_word,
                translated_word,
                target_lang,
                proficiency_level=req.proficiency_level,
                use_llm=False,
            )

        _cache_translation(source_word, source_lang, target_lang, translated_word, ipa)

        return FlashcardResponse(
            flashcard_id=f"fc_{uuid.uuid4().hex[:8]}",
            source_word=source_word,
            translated_word=translated_word,
            ipa=ipa,
            example_sentence=example_sentence,
            example_translation=example_translation,
            translation_source="translation_memory",
            cached=True,
        )

    source_candidates = _source_word_candidates(source_word)

    cached_translation = None
    cached_source = req.source_word
    for candidate in source_candidates:
        cached_translation = _lookup_translation(candidate, req.source_lang, req.target_lang)
        if cached_translation is not None:
            cached_source = candidate
            break

    if cached_translation is not None:
        translated_word, ipa = cached_translation
        if cached_source != req.source_word:
            _cache_translation(req.source_word, req.source_lang, req.target_lang, translated_word, ipa)

        example_sentence, example_translation = _build_example_sentence(
            cached_source,
            translated_word,
            req.target_lang,
            proficiency_level=req.proficiency_level,
            use_llm=False,
        )
        return FlashcardResponse(
            flashcard_id=f"fc_{uuid.uuid4().hex[:8]}",
            source_word=req.source_word,
            translated_word=translated_word,
            ipa=ipa,
            example_sentence=example_sentence,
            example_translation=example_translation,
            translation_source="translation_memory",
            cached=True,
        )

    try:
        translated_word, ipa, example_sentence, example_translation, source = _fetch_translation(
            req.source_word,
            req.source_lang,
            req.target_lang,
            req.proficiency_level,
        )
    except TranslationError as exc:
        if raise_on_error:
            raise
        print(f"[Translation] {exc}")
        normalized_source = source_candidates[0] if source_candidates else req.source_word.lower()
        translated_word = _EMERGENCY_TRANSLATION_MAP.get((normalized_source, req.target_lang), normalized_source)
        ipa = _resolve_ipa(translated_word, req.target_lang, "n/a", allow_llm=False)
        example_sentence, example_translation = _build_example_sentence(
            normalized_source,
            translated_word,
            req.target_lang,
            proficiency_level=req.proficiency_level,
            use_llm=False,
        )
        source = "error_fallback"

    if interactive and source != "error_fallback":
        lang_names = {
            "es": "Spanish",
            "fr": "French",
            "ja": "Japanese",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "ar": "Arabic",
            "zh": "Chinese",
            "ko": "Korean",
            "ru": "Russian",
            "nl": "Dutch",
            "tr": "Turkish",
        }
        lang_name = lang_names.get(req.target_lang, req.target_lang)
        print("\nTranslation verification")
        print(f"  English: {req.source_word}")
        print(f"  {lang_name}: {translated_word}")

        confirm = input("Is this translation correct? [Y/n]: ").strip().lower()
        if confirm == "n":
            corrected = input(f"Enter correct {lang_name} translation: ").strip()
            if corrected:
                translated_word = corrected
                ipa = _resolve_ipa(corrected, req.target_lang, "n/a")
                example_sentence, example_translation = _build_example_sentence(
                    req.source_word,
                    corrected,
                    req.target_lang,
                    proficiency_level=req.proficiency_level,
                    use_llm=True,
                )
                source = "human_verified"
            else:
                print("  Empty correction received; keeping the generated translation.")
        else:
            source = "human_verified"

    _cache_translation(req.source_word, req.source_lang, req.target_lang, translated_word, ipa)

    return FlashcardResponse(
        flashcard_id=f"fc_{uuid.uuid4().hex[:8]}",
        source_word=req.source_word,
        translated_word=translated_word,
        ipa=ipa,
        example_sentence=example_sentence,
        example_translation=example_translation,
        translation_source=source,
        cached=False,
    )
