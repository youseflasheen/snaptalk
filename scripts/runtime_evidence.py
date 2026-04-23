#!/usr/bin/env python3
"""Generate runtime evidence report for architecture defense and CI traceability."""

from __future__ import annotations

import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import settings
from app.main import app

OUTPUT = Path("docs/RUNTIME_VERIFICATION.md")


def _route_rows() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)
        if not methods or not path:
            continue
        method_text = ",".join(sorted(m for m in methods if m not in {"HEAD", "OPTIONS"}))
        rows.append((method_text, path))
    rows.sort(key=lambda item: item[1])
    return rows


def _legacy_route_present(paths: list[str]) -> bool:
    legacy = {"/v1/vision/detect", "/v1/pipeline/snap-learn"}
    return any(path in legacy for path in paths)


def _benchmark(client: TestClient, path: str, method: str = "GET", payload: dict | None = None, loops: int = 15) -> dict:
    elapsed_ms: list[float] = []
    status_codes: list[int] = []

    for _ in range(loops):
        start = time.perf_counter()
        if method == "GET":
            resp = client.get(path)
        else:
            resp = client.post(path, json=payload)
        end = time.perf_counter()
        elapsed_ms.append((end - start) * 1000.0)
        status_codes.append(resp.status_code)

    return {
        "loops": loops,
        "codes": sorted(set(status_codes)),
        "min_ms": round(min(elapsed_ms), 2),
        "p50_ms": round(statistics.median(elapsed_ms), 2),
        "p95_ms": round(sorted(elapsed_ms)[int(0.95 * (loops - 1))], 2),
        "max_ms": round(max(elapsed_ms), 2),
    }


def _bool_str(value: bool) -> str:
    return "yes" if value else "no"


def main() -> None:
    client = TestClient(app)

    routes = _route_rows()
    paths = [path for _methods, path in routes]

    topology = "active-modular-only" if not _legacy_route_present(paths) else "mixed-active-legacy"

    health_bench = _benchmark(client, "/health")
    translation_bench = _benchmark(
        client,
        "/v1/translation/flashcard",
        method="POST",
        payload={
            "user_id": "runtime-evidence",
            "object_id": "obj_rt",
            "source_word": "table",
            "source_lang": "en",
            "target_lang": "en",
            "proficiency_level": "A2",
        },
    )

    # Deterministic runtime checks for defense claims.
    flashcard = client.post(
        "/v1/translation/flashcard",
        json={
            "user_id": "runtime-evidence",
            "object_id": "obj_rt2",
            "source_word": "book",
            "source_lang": "en",
            "target_lang": "en",
            "proficiency_level": "A2",
        },
    )
    flash_body = flashcard.json() if flashcard.status_code == 200 else {}

    legacy_vision = client.post(
        "/v1/vision/detect",
        json={"image_url": "https://example.com/x.jpg", "max_objects": 1, "language": "en"},
    )
    legacy_pipeline = client.post(
        "/v1/pipeline/snap-learn",
        files={"image": ("x.jpg", b"x", "image/jpeg")},
        data={"target_lang": "es", "max_objects": "1"},
    )

    deepl_present = bool(settings.deepl_api_key.strip())
    gcloud_project_present = bool(settings.google_cloud_project_id.strip())
    gcloud_credentials_path = settings.google_cloud_credentials_path.strip()
    gcloud_credentials_file_present = bool(gcloud_credentials_path and Path(gcloud_credentials_path).exists())

    lines: list[str] = []
    lines.append("# Runtime Verification")
    lines.append("")
    lines.append(f"Generated at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## 1) Deployment Topology (Runtime-Proven)")
    lines.append("")
    lines.append(f"Resolved topology mode: {topology}")
    lines.append(f"Legacy routes mounted: {_bool_str(topology != 'active-modular-only')}")
    lines.append("")
    lines.append("Mounted routes:")
    lines.append("")
    lines.append("| Methods | Path |")
    lines.append("|---|---|")
    for methods, path in routes:
        lines.append(f"| {methods} | {path} |")

    lines.append("")
    lines.append("## 2) Quantitative Latency Benchmarks")
    lines.append("")
    lines.append("Benchmarks executed in-process with FastAPI TestClient.")
    lines.append("")
    lines.append("| Endpoint | Loops | Status codes | Min (ms) | P50 (ms) | P95 (ms) | Max (ms) |")
    lines.append("|---|---:|---|---:|---:|---:|---:|")
    lines.append(
        f"| GET /health | {health_bench['loops']} | {health_bench['codes']} | {health_bench['min_ms']} | {health_bench['p50_ms']} | {health_bench['p95_ms']} | {health_bench['max_ms']} |"
    )
    lines.append(
        f"| POST /v1/translation/flashcard (en->en) | {translation_bench['loops']} | {translation_bench['codes']} | {translation_bench['min_ms']} | {translation_bench['p50_ms']} | {translation_bench['p95_ms']} | {translation_bench['max_ms']} |"
    )

    lines.append("")
    lines.append("## 3) Legacy Endpoint Exposure Check")
    lines.append("")
    lines.append(f"POST /v1/vision/detect status: {legacy_vision.status_code}")
    lines.append(f"POST /v1/pipeline/snap-learn status: {legacy_pipeline.status_code}")

    lines.append("")
    lines.append("## 4) Translation Runtime Configuration")
    lines.append("")
    lines.append(f"settings.translation_google_mode: {settings.translation_google_mode}")
    lines.append(f"settings.translation_default_model: {settings.translation_default_model}")
    lines.append(
        f"en->en flashcard source observed: {flash_body.get('translation_source', 'n/a')} (status {flashcard.status_code})"
    )

    lines.append("")
    lines.append("## 5) Credential Presence Validation")
    lines.append("")
    lines.append("This verifies presence only, not external provider connectivity.")
    lines.append("")
    lines.append("| Credential/Input | Present |")
    lines.append("|---|---|")
    lines.append(f"| DeepL API key configured | {_bool_str(deepl_present)} |")
    lines.append(f"| Google Cloud project id configured | {_bool_str(gcloud_project_present)} |")
    lines.append(f"| Google credentials path configured | {_bool_str(bool(gcloud_credentials_path))} |")
    lines.append(f"| Google credentials file exists at configured path | {_bool_str(gcloud_credentials_file_present)} |")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
