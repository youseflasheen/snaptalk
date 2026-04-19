import time
import uuid
from typing import Any
from typing import Optional

import httpx

from app.core.config import settings
from app.schemas.vision import VisionDetectRequest, VisionDetectResponse, VisionObject
from app.services.vision_local import run_local_onnx
from app.utils.network_security import parse_allowed_hosts, validate_external_url


def detect_objects(req: VisionDetectRequest) -> VisionDetectResponse:
    start = time.time()

    validate_external_url(
        req.image_url,
        allowed_hosts=parse_allowed_hosts(settings.allowed_external_hosts),
        allow_private=settings.allow_private_network_urls,
    )

    objects, fallback_used = _run_real_pipeline(req)

    latency_ms = int((time.time() - start) * 1000)

    return VisionDetectResponse(
        request_id=f"req_{uuid.uuid4().hex[:8]}",
        model_version=settings.vision_model_version,
        image_width=1080,
        image_height=1920,
        objects=objects,
        latency_ms=latency_ms,
        fallback_used=fallback_used,
    )


def production_replacement_note() -> str:
    return (
        "Vision uses one real pipeline: YOLO-World -> salient ranking -> MobileSAM -> RAM++. "
        "If dependent services are unavailable, a deterministic fallback sample is returned."
    )


def _run_real_pipeline(req: VisionDetectRequest) -> tuple[list[VisionObject], bool]:
    if settings.vision_backend.lower() == "local_onnx":
        local_objects = run_local_onnx(req.image_url, req.max_objects)
        if local_objects is not None:
            return [VisionObject(**obj) for obj in local_objects], False
        return _fallback_objects(req.max_objects), True

    try:
        with httpx.Client(timeout=settings.vision_timeout_seconds) as client:
            detections = _call_yolo(client, req)
            ranked = _rank_salient(detections, req.max_objects)
            objects = _segment_and_tag(client, req, ranked)
            return objects, False
    except Exception:
        return _fallback_objects(req.max_objects), True


def _post_json(client: httpx.Client, url: str, payload: dict, attempts: int = 2) -> dict:
    last_error: Optional[Exception] = None
    for _ in range(attempts):
        try:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # pragma: no cover
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError("Request failed")


def _call_yolo(client: httpx.Client, req: VisionDetectRequest) -> list[dict]:
    payload = {
        "image_url": req.image_url,
        "max_objects": max(req.max_objects * 2, 10),
        "language": req.language,
    }
    data = _post_json(client, settings.yolo_service_url, payload)
    detections = data.get("detections", [])
    return [d for d in detections if _valid_bbox(d.get("bbox"))]


def _valid_bbox(bbox: Any) -> bool:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        return x2 > x1 and y2 > y1
    except Exception:
        return False


def _rank_salient(detections: list[dict], top_n: int) -> list[dict]:
    scored: list[dict] = []
    for det in detections:
        score = (
            0.4 * float(det.get("confidence", 0.0))
            + 0.3 * float(det.get("center_proximity", 0.0))
            + 0.3 * float(det.get("area_ratio", 0.0))
        )
        det["saliency_score"] = score
        scored.append(det)

    scored.sort(key=lambda x: x.get("saliency_score", 0.0), reverse=True)
    return scored[:top_n]


def _segment_and_tag(client: httpx.Client, req: VisionDetectRequest, ranked: list[dict]) -> list[VisionObject]:
    out: list[VisionObject] = []

    for i, det in enumerate(ranked, start=1):
        bbox = [int(x) for x in det.get("bbox", [0, 0, 1, 1])]

        seg_data = _post_json(
            client,
            settings.mobilesam_service_url,
            {"image_url": req.image_url, "bbox": bbox},
        )
        polygon = _normalize_polygon(seg_data.get("polygon", []), bbox)

        tag_data = _post_json(
            client,
            settings.rampp_service_url,
            {"image_url": req.image_url, "bbox": bbox},
        )
        tags = tag_data.get("tags", [])

        canonical_tag = tags[-1] if tags else det.get("label", "unknown")

        out.append(
            VisionObject(
                object_id=f"obj_{i}",
                bbox=bbox,
                polygon=polygon,
                detector_label=str(det.get("label", "unknown")),
                canonical_tag=str(canonical_tag),
                confidence=float(det.get("confidence", 0.0)),
                source="rampp",
            )
        )

    return out


def _normalize_polygon(polygon: Any, bbox: list[int]) -> list[list[int]]:
    if not isinstance(polygon, list) or len(polygon) < 3:
        x1, y1, x2, y2 = bbox
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

    normalized: list[list[int]] = []
    for point in polygon:
        if not isinstance(point, list) or len(point) != 2:
            continue
        try:
            normalized.append([int(point[0]), int(point[1])])
        except Exception:
            continue

    if len(normalized) < 3:
        x1, y1, x2, y2 = bbox
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    return normalized


def _fallback_objects(max_objects: int) -> list[VisionObject]:
    sample = [
        VisionObject(
            object_id="obj_1",
            bbox=[120, 300, 420, 780],
            polygon=[[130, 310], [410, 320], [405, 770], [125, 760]],
            detector_label="apple",
            canonical_tag="apple",
            confidence=0.92,
            source="fallback",
        ),
        VisionObject(
            object_id="obj_2",
            bbox=[500, 280, 860, 900],
            polygon=[[520, 300], [850, 320], [840, 880], [510, 890]],
            detector_label="book",
            canonical_tag="book",
            confidence=0.86,
            source="fallback",
        ),
    ]
    return sample[:max_objects]
