from types import SimpleNamespace
from pathlib import Path

import scripts.snap_learn as snap_learn
from app.core.config import settings


def test_option_two_cycles_languages_for_same_object(monkeypatch) -> None:
    obj = SimpleNamespace(canonical_tag="ball", confidence=0.9)
    result = SimpleNamespace(objects=[obj], image_width=320, image_height=240, total_objects=1)

    monkeypatch.setattr(snap_learn, "clear_screen", lambda: None)
    monkeypatch.setattr(snap_learn, "print_header", lambda: None)
    monkeypatch.setattr(snap_learn, "load_image", lambda _path: b"image-bytes")
    monkeypatch.setattr(snap_learn, "detect_objects", lambda _image_bytes: result)
    monkeypatch.setattr(snap_learn, "select_session_languages", lambda: ("ar", "es"))

    display_calls = {"count": 0}

    def _display_objects(_result):
        display_calls["count"] += 1
        return {1: obj}

    monkeypatch.setattr(snap_learn, "display_objects", _display_objects)
    monkeypatch.setattr(snap_learn, "select_object", lambda _objects: obj)
    monkeypatch.setattr(snap_learn, "translate_label_to_native", lambda _word, _lang: "كرة")
    monkeypatch.setattr(
        snap_learn,
        "display_languages",
        lambda *args, **kwargs: {1: "ru", 2: "de"},
    )

    selected_langs = iter(["ru", "de"])
    monkeypatch.setattr(snap_learn, "select_language", lambda _lang_map, **kwargs: next(selected_langs))

    cycle_calls: list[str] = []
    monkeypatch.setattr(
        snap_learn,
        "run_target_language_cycle",
        lambda selected_obj, native_lang, target_lang, native_object_label: cycle_calls.append(target_lang) or True,
    )

    yes_no_answers = iter([
        True,   # translate same word into another language? (after es)
        True,   # translate same word into another language? (after ru)
        False,  # translate same word into another language? (after de)
        False,  # select another object?
    ])
    monkeypatch.setattr(snap_learn, "_prompt_yes_no", lambda _prompt, default: next(yes_no_answers))
    monkeypatch.setattr(snap_learn, "save_artifacts_info", lambda: None)
    monkeypatch.setattr("sys.argv", ["snap_learn.py", "--image", "fake.jpg"])

    snap_learn.main()

    assert display_calls["count"] == 1
    assert cycle_calls == ["es", "ru", "de"]


def test_display_objects_uses_native_label(monkeypatch, capsys) -> None:
    obj = SimpleNamespace(canonical_tag="Rubik's cube", confidence=0.67)
    result = SimpleNamespace(objects=[obj], total_objects=1)

    snap_learn._set_session_native_language("ja")
    monkeypatch.setattr(snap_learn, "translate_label_to_native", lambda _word, _lang: "ルービックキューブ")

    snap_learn.display_objects(result)
    out = capsys.readouterr().out

    assert "ルービックキューブ" in out


def test_prune_audio_storage_keeps_only_requested_files(tmp_path: Path) -> None:
    original_tts_output_dir = settings.tts_output_dir
    settings.tts_output_dir = str(tmp_path)

    old_file = tmp_path / "aud_old.wav"
    keep_file = tmp_path / "aud_keep.wav"
    old_file.write_bytes(b"old")
    keep_file.write_bytes(b"keep")

    try:
        snap_learn._prune_audio_storage([str(keep_file)])
    finally:
        settings.tts_output_dir = original_tts_output_dir

    assert keep_file.exists()
    assert not old_file.exists()
