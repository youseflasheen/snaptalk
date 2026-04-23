from pathlib import Path
import re

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app


client = TestClient(app)


def test_health() -> None:
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert "env" not in body


def test_legacy_routes_are_not_exposed() -> None:
    vision = client.post(
        "/v1/vision/detect",
        json={
            "image_url": "https://example.com/i.jpg",
            "max_objects": 2,
            "language": "en",
        },
    )
    assert vision.status_code == 404

    legacy_pipeline = client.post(
        "/v1/pipeline/snap-learn",
        files={"image": ("x.jpg", b"x", "image/jpeg")},
        data={"target_lang": "es", "max_objects": "1"},
    )
    assert legacy_pipeline.status_code == 404


def test_pipeline_vlm_rejects_oversized_upload() -> None:
    original_limit = settings.max_upload_image_bytes
    settings.max_upload_image_bytes = 1024

    try:
        oversized = b"\x89PNG\r\n\x1a\n" + (b"0" * 2048)
        res = client.post(
            "/v1/pipeline/snap-learn-vlm",
            files={"image": ("big.png", oversized, "image/png")},
            data={"target_lang": "es", "max_objects": "2"},
        )
        assert res.status_code == 413
    finally:
        settings.max_upload_image_bytes = original_limit


def test_pipeline_vlm_contract_rejects_non_image() -> None:
    res = client.post(
        "/v1/pipeline/snap-learn-vlm",
        files={"image": ("bad.txt", b"hello", "text/plain")},
        data={"target_lang": "es", "max_objects": "2"},
    )
    assert res.status_code == 400


def test_translation_cache_roundtrip(tmp_path: Path) -> None:
    settings.translation_db_path = str(tmp_path / "translation_memory.db")

    payload = {
        "user_id": "u1",
        "object_id": "o1",
        "source_word": "table",
        "source_lang": "en",
        "target_lang": "es",
        "proficiency_level": "A2",
    }

    first = client.post("/v1/translation/flashcard", json=payload)
    assert first.status_code == 200
    first_body = first.json()

    second = client.post("/v1/translation/flashcard", json=payload)
    assert second.status_code == 200
    second_body = second.json()

    # First request may use fallback, second should be cached in translation memory.
    assert first_body["translated_word"] == second_body["translated_word"]
    assert second_body["translation_source"] == "translation_memory"


def test_translation_same_language_en_to_en_returns_identity(tmp_path: Path) -> None:
    settings.translation_db_path = str(tmp_path / "translation_memory.db")

    payload = {
        "user_id": "u1",
        "object_id": "o_same",
        "source_word": "Rubik's cube",
        "source_lang": "en",
        "target_lang": "en",
        "proficiency_level": "A2",
    }

    res = client.post("/v1/translation/flashcard", json=payload)
    assert res.status_code == 200
    body = res.json()

    assert body["translated_word"] == "Rubik's cube"
    assert body["translation_source"] == "translation_memory"
    assert body["example_sentence"].strip()


def test_tts_generates_audio_file(tmp_path: Path) -> None:
    settings.tts_output_dir = str(tmp_path / "audio")
    settings.public_base_url = "http://127.0.0.1:8000"

    res = client.post(
        "/v1/speech/tts",
        json={
            "text": "hola mundo",
            "lang_code": "es",
            "voice": "default",
            "speed": 1.0,
        },
    )
    assert res.status_code == 200
    body = res.json()

    audio_id = body["audio_id"]
    expected = Path(settings.tts_output_dir) / f"{audio_id}.wav"
    assert expected.exists()
    assert body["audio_url"].endswith(f"{audio_id}.wav")


def test_pronunciation_response_shape() -> None:
    res = client.post(
        "/v1/speech/pronunciation",
        json={
            "user_id": "u1",
            "reference_text": "hola",
            "reference_phonemes": ["h", "o", "l", "a"],
            "lang_code": "es",
            "audio_url": "https://example.com/a.wav",
        },
    )
    assert res.status_code == 200
    body = res.json()

    assert 0 <= body["overall_score"] <= 1
    assert len(body["per_phoneme"]) == 4
    assert "alignment" in body


def test_pronunciation_response_includes_levels() -> None:
    res = client.post(
        "/v1/speech/pronunciation",
        json={
            "user_id": "u1",
            "reference_text": "hello",
            "reference_phonemes": ["h", "e", "l", "o"],
            "lang_code": "en",
            "audio_url": "https://example.com/a.wav",
        },
    )
    assert res.status_code == 200
    body = res.json()

    assert "overall_level" in body
    assert body["overall_level"] in ["red", "orange", "green"]
    for phoneme in body["per_phoneme"]:
        assert "level" in phoneme
        assert phoneme["level"] in ["red", "orange", "green"]


def test_overall_level_is_gated_by_phoneme_levels() -> None:
    from app.services.pronunciation.service import _resolve_overall_level

    assert _resolve_overall_level(0.86, ["green", "green", "green"]) == "green"
    assert _resolve_overall_level(0.86, ["green", "orange", "green"]) == "orange"
    assert _resolve_overall_level(0.86, ["green", "red", "green"]) == "orange"
    assert _resolve_overall_level(0.92, ["green", "red", "red", "green"]) == "red"
    assert _resolve_overall_level(0.92, ["green", "orange", "green", "green"]) == "green"
    assert _resolve_overall_level(0.52, ["green", "green", "green"]) == "red"


def test_score_pronunciation_from_audio_bytes_simulation_mode() -> None:
    from app.schemas.speech import PronunciationRequest
    from app.services.pronunciation import service as svc

    original_mode = settings.pronunciation_mode
    original_local_enabled = settings.pronunciation_local_enabled

    settings.pronunciation_mode = "simulation"
    settings.pronunciation_local_enabled = False

    req = PronunciationRequest(
        user_id="u1",
        reference_text="hola",
        reference_phonemes=["h", "o", "l", "a"],
        lang_code="es",
        audio_url="local-audio://attempt.wav",
    )

    try:
        resp = svc.score_pronunciation_from_audio_bytes(req, b"not-a-real-wav")
    finally:
        settings.pronunciation_mode = original_mode
        settings.pronunciation_local_enabled = original_local_enabled

    assert 0 <= resp.overall_score <= 1
    assert resp.overall_level in ["red", "orange", "green"]
    assert len(resp.per_phoneme) == len(req.reference_phonemes)


def test_warmup_local_pronunciation_models_disabled() -> None:
    from app.services.pronunciation.service import warmup_local_pronunciation_models

    original_local_enabled = settings.pronunciation_local_enabled
    settings.pronunciation_local_enabled = False

    try:
        ok, message = warmup_local_pronunciation_models()
    finally:
        settings.pronunciation_local_enabled = original_local_enabled

    assert ok is False
    assert "disabled" in message.lower()


def test_tts_uses_edge_engine(tmp_path: Path) -> None:
    settings.tts_output_dir = str(tmp_path / "audio")
    settings.public_base_url = "http://127.0.0.1:8000"

    res = client.post(
        "/v1/speech/tts",
        json={
            "text": "hello world",
            "lang_code": "en",
            "voice": "default",
            "speed": 1.0,
        },
    )
    assert res.status_code == 200
    body = res.json()

    assert body["engine"] in ["edge", "silence"]
    assert "piper" not in body["engine"].lower()


def test_translation_with_new_sources(tmp_path: Path) -> None:
    settings.translation_db_path = str(tmp_path / "translation_memory.db")

    payload = {
        "user_id": "u1",
        "object_id": "o1",
        "source_word": "table",
        "source_lang": "en",
        "target_lang": "es",
        "proficiency_level": "A2",
    }

    res = client.post("/v1/translation/flashcard", json=payload)
    assert res.status_code == 200
    body = res.json()

    valid_sources = [
        "translation_memory",
        "google_cloud",
        "google",
        "deepl",
        "mymemory",
        "ollama",
        "human_verified",
        "error_fallback",
    ]
    assert body["translation_source"] in valid_sources


def test_translation_typo_source_still_returns_word(tmp_path: Path) -> None:
    settings.translation_db_path = str(tmp_path / "translation_memory.db")

    payload = {
        "user_id": "u1",
        "object_id": "o1",
        "source_word": "hoddie",
        "source_lang": "en",
        "target_lang": "ar",
        "proficiency_level": "A2",
    }

    res = client.post("/v1/translation/flashcard", json=payload)
    assert res.status_code == 200
    body = res.json()

    assert body["translated_word"]
    assert body["translated_word"] != "hoddie_ar"
    assert not body["translated_word"].endswith("_ar")


def test_source_candidates_prioritize_corrected_word() -> None:
    from app.services.translation.service import _source_word_candidates

    candidates = _source_word_candidates("hoddie")
    assert candidates[0] == "hoodie"
    assert "hoddie" in candidates


def test_translation_quality_rejects_placeholder_suffix() -> None:
    from app.services.translation.service import _translation_looks_low_quality

    assert _translation_looks_low_quality("hoddie", "hoddie_ar", "ar") is True
    assert _translation_looks_low_quality("hoodie", "هودي", "ar") is False


def test_known_term_guard_coerces_bad_output() -> None:
    from app.services.translation.service import _coerce_known_term_translation

    translated, used_override = _coerce_known_term_translation("hoddie", "ar", "زُنط")
    assert used_override is True
    assert translated == "هودي"

    translated_ok, used_override_ok = _coerce_known_term_translation("hoodie", "fr", "sweat à capuche")
    assert used_override_ok is False
    assert translated_ok == "sweat à capuche"


def test_fetch_translation_prefers_deepl(monkeypatch) -> None:
    from app.services.translation import service as svc

    original_mode = settings.translation_google_mode
    settings.translation_google_mode = "official_with_fallback"

    monkeypatch.setattr(svc, "_deepl_translate", lambda *_: ("sudadera", "n/a"))
    monkeypatch.setattr(svc, "_google_cloud_translate", lambda *_: ("هودي", "n/a"))
    monkeypatch.setattr(svc, "_google_translate", lambda *_: ("bad", "n/a"))
    monkeypatch.setattr(svc, "_mymemory_translate", lambda *_: None)
    monkeypatch.setattr(svc, "_ollama_translate", lambda *_: None)
    monkeypatch.setattr(svc, "_resolve_ipa", lambda *_args, **_kwargs: "h.u.d.i")
    monkeypatch.setattr(svc, "_build_example_sentence", lambda *_args, **_kwargs: ("X", "Y"))

    try:
        translated, _ipa, _sentence, _meaning, source = svc._fetch_translation("hoddie", "en", "es", "A2")
    finally:
        settings.translation_google_mode = original_mode

    assert translated == "sudadera"
    assert source == "deepl"


def test_ipa_refusal_text_is_rejected() -> None:
    from app.services.translation.service import _ipa_looks_low_quality

    refusal = "I can't provide information or guidance on illegal or harmful activities."
    assert _ipa_looks_low_quality(refusal, "hoddie_ar", "ar") is True
    assert _ipa_looks_low_quality("h.u.d.i", "هودي", "ar") is False


def test_translation_ipa_not_cyrillic_echo(tmp_path: Path) -> None:
    settings.translation_db_path = str(tmp_path / "translation_memory.db")

    payload = {
        "user_id": "u1",
        "object_id": "o1",
        "source_word": "table",
        "source_lang": "en",
        "target_lang": "ru",
        "proficiency_level": "A2",
    }

    res = client.post("/v1/translation/flashcard", json=payload)
    assert res.status_code == 200
    body = res.json()

    cyrillic = re.compile(r"[\u0400-\u04FF]")
    if cyrillic.search(body["translated_word"]):
        assert not cyrillic.search(body["ipa"]), "IPA should not echo Cyrillic script"


def test_translation_example_sentence_in_target_language(tmp_path: Path) -> None:
    settings.translation_db_path = str(tmp_path / "translation_memory.db")

    payload = {
        "user_id": "u1",
        "object_id": "o1",
        "source_word": "table",
        "source_lang": "en",
        "target_lang": "es",
        "proficiency_level": "A2",
    }

    res = client.post("/v1/translation/flashcard", json=payload)
    assert res.status_code == 200
    body = res.json()

    assert body["example_sentence"].strip()
    assert not body["example_sentence"].strip().lower().startswith("this is ")
    assert body["example_translation"].strip()
    assert not body["example_translation"].strip().startswith("[")


def test_snap_learn_returns_semantic_labels(tmp_path: Path) -> None:
    """Integration: YOLO model must return real semantic labels, never generic placeholders.

    Skipped automatically when:
    - ultralytics is not installed (pip install ultralytics)
    - data/sample_real.jpg is not present
    """
    pytest.importorskip("ultralytics")

    sample = Path("data/sample_real.jpg")
    if not sample.exists():
        pytest.skip("data/sample_real.jpg not found – place a real photo there to run this test")

    settings.tts_output_dir = str(tmp_path / "audio")

    with sample.open("rb") as f:
        res = client.post(
            "/v1/pipeline/snap-learn-vlm",
            files={"image": ("sample_real.jpg", f, "image/jpeg")},
            data={"target_lang": "es", "max_objects": "5"},
        )

    assert res.status_code == 200
    body = res.json()
    assert "objects" in body
    assert "image_width" in body and body["image_width"] > 0

    for obj in body["objects"]:
        # Must NOT be a generic placeholder produced by the old heuristic
        tag = obj["canonical_tag"]
        assert not tag.startswith("object"), (
            f"Generic label '{tag}' returned – YOLO model must produce semantic COCO class names."
        )
        assert tag != "unknown", "Label should not be 'unknown' for a real detected object."
        # Polygon must be a real shape with at least 3 vertices
        assert len(obj["polygon"]) >= 3, f"Polygon for '{tag}' has fewer than 3 points."
        # Translated word must be non-empty (translation memory or LLM fallback)
        assert obj["translated_word"], f"translated_word is empty for '{tag}'."

    # Visual artifacts should have been written
    assert Path("data/artifacts/polygon_overlay.jpg").exists(), (
        "polygon_overlay.jpg was not saved – check _save_artifacts in snap_learn_service.py"
    )
