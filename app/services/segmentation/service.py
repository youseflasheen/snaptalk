from __future__ import annotations

import base64
import glob
import os
import threading
from typing import Any

import cv2
import numpy as np

from app.core.config import settings

_sam: Any = None
_sam_lock = threading.Lock()

_MASK_COLORS = [
    (0, 200, 80),
    (0, 120, 255),
    (255, 60, 60),
    (200, 0, 255),
    (0, 220, 220),
]


def get_sam() -> Any:
    global _sam
    if _sam is None:
        with _sam_lock:
            if _sam is None:
                from ultralytics import SAM

                _sam = SAM("mobile_sam.pt")
    return _sam


def simplify_polygon(coords: np.ndarray, epsilon_ratio: float = 0.015) -> list[list[int]]:
    pts = coords.reshape(-1, 1, 2).astype(np.int32)
    arc = cv2.arcLength(pts, True)
    epsilon = max(1.0, epsilon_ratio * arc)
    approx = cv2.approxPolyDP(pts, epsilon, True)
    simplified = [[int(p[0][0]), int(p[0][1])] for p in approx]
    return simplified if len(simplified) >= 3 else []


def bbox_polygon(bbox: list[int]) -> list[list[int]]:
    x1, y1, x2, y2 = bbox
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def segment_with_sam(image_rgb: np.ndarray, bbox: list[int]) -> list[list[int]]:
    try:
        model = get_sam()
        results = model(image_rgb, bboxes=[bbox], verbose=False)
        if not results:
            return bbox_polygon(bbox)

        masks_obj = results[0].masks
        if masks_obj is None or len(masks_obj.xy) == 0:
            return bbox_polygon(bbox)

        coords = masks_obj.xy[0]
        if len(coords) >= 3:
            poly = simplify_polygon(coords)
            if poly:
                return poly
    except Exception:
        pass
    return bbox_polygon(bbox)


def extract_masked_crop(
    image_bgr: np.ndarray,
    polygon: list[list[int]],
    transparent: bool = False,
) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)

    x_coords = [p[0] for p in polygon]
    y_coords = [p[1] for p in polygon]
    x1, x2 = max(0, min(x_coords)), min(w, max(x_coords))
    y1, y2 = max(0, min(y_coords)), min(h, max(y_coords))

    if transparent:
        bgra = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = mask
        crop = bgra[y1:y2, x1:x2]
    else:
        masked = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
        crop = masked[y1:y2, x1:x2]

    return crop


def encode_masked_png_base64(masked_crop: np.ndarray) -> str:
    if masked_crop.size == 0:
        return ""

    max_dim = max(64, int(settings.masked_image_max_dim))
    working = masked_crop
    h, w = working.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        target = (max(1, int(w * scale)), max(1, int(h * scale)))
        working = cv2.resize(working, target, interpolation=cv2.INTER_AREA)

    encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
    ok, buffer = cv2.imencode(".png", working, encode_params)
    if not ok:
        return ""

    max_png_bytes = max(50_000, int(settings.masked_image_png_max_bytes))
    while len(buffer) > max_png_bytes and min(working.shape[:2]) > 96:
        resized = cv2.resize(
            working,
            (max(1, working.shape[1] // 2), max(1, working.shape[0] // 2)),
            interpolation=cv2.INTER_AREA,
        )
        ok, buffer = cv2.imencode(".png", resized, encode_params)
        if not ok:
            return ""
        working = resized

    if len(buffer) > max_png_bytes:
        return ""

    return base64.b64encode(buffer).decode("utf-8")


def save_artifacts(image_bgr: np.ndarray, detections: list[dict]) -> None:
    try:
        output_dir = "./data/artifacts"
        os.makedirs(output_dir, exist_ok=True)

        for old_file in glob.glob(os.path.join(output_dir, "*.jpg")) + glob.glob(os.path.join(output_dir, "*.png")):
            try:
                os.remove(old_file)
            except Exception:
                pass

        result = image_bgr.copy()

        for i, det in enumerate(detections):
            tag = det["canonical_tag"]
            x1, y1, x2, y2 = det["bbox"]
            polygon = det.get("polygon", [])
            color = _MASK_COLORS[i % len(_MASK_COLORS)]

            if len(polygon) >= 3:
                pts = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
                fill_layer = result.copy()
                cv2.fillPoly(fill_layer, [pts], color)
                cv2.addWeighted(fill_layer, 0.35, result, 0.65, 0, result)
                cv2.polylines(result, [pts], True, color, 2)

            cv2.rectangle(result, (x1, y1), (x2, y2), color, 1)
            label_bg_y = max(0, y1 - 28)
            text = f"{tag} ({det.get('confidence', 0):.2f})"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(result, (x1, label_bg_y), (x1 + tw + 6, label_bg_y + th + 8), color, -1)
            cv2.putText(
                result,
                text,
                (x1 + 3, label_bg_y + th + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            crop = image_bgr[y1:y2, x1:x2]
            if crop.size > 0:
                cv2.imwrite(os.path.join(output_dir, f"{det['object_id']}_bbox_crop.jpg"), crop)

            if len(polygon) >= 3:
                masked_crop = extract_masked_crop(image_bgr, polygon, transparent=True)
                cv2.imwrite(os.path.join(output_dir, f"{det['object_id']}_masked.png"), masked_crop)

        cv2.imwrite(os.path.join(output_dir, "polygon_overlay.jpg"), result)
    except Exception:
        pass
