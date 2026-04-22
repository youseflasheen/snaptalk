"""
Snap & Learn pipeline — Option A: full spec
============================================
Step 1  YOLO-World  (open-vocabulary detection)  → bounding boxes + detector labels
          Uses built-in LVIS 1,203-category vocabulary by default (maximum coverage).
          Override via data/yolo_world_vocab.txt only to restrict to a specific domain.
Step 2  SOR         (Salient Object Ranking)      → top-N ranked objects
Step 3  MobileSAM   (segmentation via bbox prompt) → binary masks → simplified polygons
Step 4  RAM++       (fine-grained semantic tagging) → leaf-level canonical label per crop
          4,500+ category vocabulary; output is the canonical_tag in the API response.
Step 5  Translation (SQLite TM → LLM fallback)    → flashcard data
Step 6  Artifacts   (overlay image + per-object crops saved to data/artifacts/)

Model weight downloads (all automatic, cached to ~/.cache/):
  yolov8s-worldv2.pt          ~28 MB   (ultralytics HuggingFace hub)
  mobile_sam.pt               ~40 MB   (ultralytics HuggingFace hub)
  ram_plus_plus_swin_large_... ~1.4 GB  (xinyu1205/recognize-anything-plus-model HF hub)

CPU latency notes:
  YOLO-World + SAM: ~1–3 s per image
  RAM++ per crop:   ~5–15 s on CPU  (acceptable for dev; intended for GPU in production)
  RAM++ falls back to YOLO-World detector label if unavailable or errored.
"""

from __future__ import annotations

import base64
import os
import threading
import warnings
from typing import Any, Optional

import cv2
import numpy as np

from app.core.config import settings
from app.schemas.pipeline import SnapLearnObject, SnapLearnResponse
from app.schemas.translation import FlashcardRequest
from app.services.translation.service import build_flashcard

# ──────────────────────────────────────────────────────────────────────────────
# Lazy-loaded model singletons
# ──────────────────────────────────────────────────────────────────────────────
_yolo_world: Any = None   # ultralytics.YOLOWorld
_sam: Any = None           # ultralytics.SAM  (MobileSAM weights)
_ram_model: Any = None     # recognize_anything ram_plus_plus
_ram_transform: Any = None
_yolo_world_lock = threading.Lock()
_sam_lock = threading.Lock()
_ram_lock = threading.Lock()


def _get_yolo_world() -> Any:
    global _yolo_world
    if _yolo_world is None:
        with _yolo_world_lock:
            if _yolo_world is None:
                try:
                    from ultralytics import YOLOWorld
                except ImportError as exc:
                    raise RuntimeError(
                        "ultralytics >= 8.3.0 is required.  Install with: pip install ultralytics"
                    ) from exc

                model = YOLOWorld("yolov8s-worldv2.pt")  # ~28 MB, auto-downloaded on first use
                # Default: no set_classes() call → model uses its full built-in LVIS 1,203-category
                # vocabulary automatically, giving maximum detection coverage.
                # set_classes() is only invoked when the team provides a custom override file,
                # e.g. to restrict detection to a specific product domain.
                custom_vocab = _load_custom_vocab()
                if custom_vocab:
                    model.set_classes(custom_vocab)
                _yolo_world = model
    return _yolo_world


def _get_sam() -> Any:
    global _sam
    if _sam is None:
        with _sam_lock:
            if _sam is None:
                try:
                    from ultralytics import SAM
                except ImportError as exc:
                    raise RuntimeError(
                        "ultralytics >= 8.3.0 is required.  Install with: pip install ultralytics"
                    ) from exc
                _sam = SAM("mobile_sam.pt")  # ~40 MB, auto-downloaded on first use
    return _sam


def _get_ram() -> tuple[Any, Any]:
    """
    Returns (model, transform).
    Returns (None, None) when recognize-anything is not installed
    or the weight download fails.  The pipeline continues with the YOLO-World label
    as the canonical_tag fallback.
    """
    global _ram_model, _ram_transform
    if _ram_model is None:
        with _ram_lock:
            if _ram_model is None:
                try:
                    import torch
                    from torchvision import transforms as T
                    from huggingface_hub import hf_hub_download
                    from ram.models.ram_plus import ram_plus

                    weights_path = hf_hub_download(
                        repo_id="xinyu1205/recognize-anything-plus-model",
                        filename="ram_plus_swin_large_14m.pth",
                    )

                    model = ram_plus(
                        pretrained=weights_path,
                        image_size=384,
                        vit="swin_l",
                    )
                    model.eval()

                    # Standard ImageNet normalisation used by RAM++ training
                    transform = T.Compose([
                        T.Resize((384, 384)),
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])

                    _ram_model = model
                    _ram_transform = transform
                except Exception as exc:
                    warnings.warn(
                        f"RAM++ could not be loaded ({exc}). "
                        "Install with: pip install recognize-anything huggingface_hub  "
                        "Canonical tag will fall back to YOLO-World detector label.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    return None, None
    return _ram_model, _ram_transform


# ──────────────────────────────────────────────────────────────────────────────
# YOLO-World vocabulary (custom override — normally left empty)
# ──────────────────────────────────────────────────────────────────────────────

def _load_custom_vocab() -> list[str]:
    """
    Returns a custom category list from data/yolo_world_vocab.txt, or [] if the file
    is absent or empty.  Returning [] causes YOLO-World to use its full built-in
    LVIS 1,203-category vocabulary — the broadest detection coverage available.
    Only populate the file when you deliberately want to restrict detection to a
    smaller domain (e.g. a specific retail product catalogue).
    """
    path = getattr(settings, "yolo_world_vocab_path", "./data/yolo_world_vocab.txt")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    return lines


# ──────────────────────────────────────────────────────────────────────────────
# SOR: Salient Object Ranking
# score = 0.4 * confidence + 0.3 * center_proximity + 0.3 * area_ratio
# ──────────────────────────────────────────────────────────────────────────────

def _sor_score(bbox: list[int], confidence: float, img_w: int, img_h: int) -> float:
    x1, y1, x2, y2 = bbox
    area = max(0.0, (x2 - x1) * (y2 - y1))
    area_ratio = min(1.0, area / float(img_w * img_h))
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    max_dist = ((img_w / 2.0) ** 2 + (img_h / 2.0) ** 2) ** 0.5
    dist = ((cx - img_w / 2.0) ** 2 + (cy - img_h / 2.0) ** 2) ** 0.5
    center_prox = max(0.0, 1.0 - dist / max_dist)
    return 0.4 * confidence + 0.3 * center_prox + 0.3 * area_ratio


# ──────────────────────────────────────────────────────────────────────────────
# Polygon helpers
# ──────────────────────────────────────────────────────────────────────────────

def _simplify_polygon(coords: np.ndarray, epsilon_ratio: float = 0.015) -> list[list[int]]:
    """Douglas-Peucker simplification on a dense (K, 2) contour array."""
    pts = coords.reshape(-1, 1, 2).astype(np.int32)
    arc = cv2.arcLength(pts, True)
    epsilon = max(1.0, epsilon_ratio * arc)
    approx = cv2.approxPolyDP(pts, epsilon, True)
    simplified = [[int(p[0][0]), int(p[0][1])] for p in approx]
    return simplified if len(simplified) >= 3 else []


def _bbox_polygon(bbox: list[int]) -> list[list[int]]:
    x1, y1, x2, y2 = bbox
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


# ──────────────────────────────────────────────────────────────────────────────
# RAM++ hierarchy filter
# ──────────────────────────────────────────────────────────────────────────────

def _pick_most_specific(tags_str: str) -> str:
    """
    RAM++ inference_ram_openset returns a ' | '-separated string of hierarchical tags, e.g.:
        "guitar | acoustic guitar | musical instrument | string instrument"

    The most specific (leaf) tag is the compound phrase with the most words.
    Falls back gracefully on unexpected formats.
    """
    if not tags_str or not tags_str.strip():
        return ""
    for sep in [" | ", "|", ",", ";"]:
        parts = [t.strip() for t in tags_str.split(sep) if t.strip()]
        if len(parts) > 1:
            return max(parts, key=lambda t: len(t.split()))
    return tags_str.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Visual artifact saving (best-effort; never crashes the API response)
# ──────────────────────────────────────────────────────────────────────────────

_MASK_COLORS = [
    (0, 200, 80),    # green
    (0, 120, 255),   # blue-orange
    (255, 60, 60),   # red
    (200, 0, 255),   # purple
    (0, 220, 220),   # cyan
]


def _save_artifacts(image_bgr: np.ndarray, detections: list[dict]) -> None:
    try:
        import glob
        output_dir = "./data/artifacts"
        os.makedirs(output_dir, exist_ok=True)

        # Clear old artifacts from previous runs
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
                # Semi-transparent filled polygon
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
                result, text, (x1 + 3, label_bg_y + th + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
            )

            # Save bbox crop (for reference)
            crop = image_bgr[y1:y2, x1:x2]
            if crop.size > 0:
                cv2.imwrite(os.path.join(output_dir, f"{det['object_id']}_bbox_crop.jpg"), crop)
            
            # Save transparent masked crop (actual segmentation)
            if len(polygon) >= 3:
                masked_crop = _extract_masked_crop(image_bgr, polygon, transparent=True)
                cv2.imwrite(os.path.join(output_dir, f"{det['object_id']}_masked.png"), masked_crop)

        cv2.imwrite(os.path.join(output_dir, "polygon_overlay.jpg"), result)
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Step 1: YOLO-World open-vocabulary detection
# ──────────────────────────────────────────────────────────────────────────────

def _detect_yolo_world(image_rgb: np.ndarray) -> list[dict]:
    """
    Returns list of {bbox, label, confidence} for all detections above threshold.
    label comes from the vocabulary set via model.set_classes() — any of the ~200+
    everyday object names, not limited to COCO-80.
    """
    model = _get_yolo_world()
    # Maximize detection: ultra-low thresholds to catch every possible object
    results = model(
        image_rgb,
        conf=0.05,      # Detect everything above 5% confidence (vs default 25%)
        iou=0.3,        # Lower IOU to avoid merging close objects (vs default 45%)
        max_det=500,    # Allow up to 500 detections for dense scenes (vs default 300)
        verbose=False
    )[0]
    boxes = results.boxes
    if boxes is None or len(boxes) == 0:
        return []

    dets: list[dict] = []
    for box in boxes:
        conf = float(box.conf[0])
        if conf < 0.05:  # Only filter completely garbage detections
            continue
        cls_idx = int(box.cls[0])
        label: str = results.names.get(cls_idx, "unknown")
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
        dets.append({"bbox": [x1, y1, x2, y2], "label": label, "confidence": conf})
    return dets


# ──────────────────────────────────────────────────────────────────────────────
# Step 3: MobileSAM segmentation (bbox-prompted)
# ──────────────────────────────────────────────────────────────────────────────

def _segment_with_sam(image_rgb: np.ndarray, bbox: list[int]) -> list[list[int]]:
    """
    Prompts MobileSAM with a single bounding box, returns a simplified polygon.
    Falls back to the 4-corner bbox polygon on any failure.
    """
    try:
        model = _get_sam()
        results = model(image_rgb, bboxes=[bbox], verbose=False)
        if not results:
            return _bbox_polygon(bbox)

        masks_obj = results[0].masks
        if masks_obj is None or len(masks_obj.xy) == 0:
            return _bbox_polygon(bbox)

        coords = masks_obj.xy[0]
        if len(coords) >= 3:
            poly = _simplify_polygon(coords)
            if poly:
                return poly
    except Exception:
        pass
    return _bbox_polygon(bbox)


# ──────────────────────────────────────────────────────────────────────────────
# Polygon masking utility
# ──────────────────────────────────────────────────────────────────────────────

def _extract_masked_crop(
    image_bgr: np.ndarray,
    polygon: list[list[int]],
    transparent: bool = False
) -> np.ndarray:
    """
    Extracts a masked crop using the segmentation polygon.

    Args:
        image_bgr: Input image in BGR format
        polygon: List of [x, y] points defining the object boundary
        transparent: If True, returns BGRA with alpha channel; if False, black background

    Returns:
        Cropped image containing only the object (black bg or transparent)
    """
    h, w = image_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)

    # Get bounding box of the polygon
    x_coords = [p[0] for p in polygon]
    y_coords = [p[1] for p in polygon]
    x1, x2 = max(0, min(x_coords)), min(w, max(x_coords))
    y1, y2 = max(0, min(y_coords)), min(h, max(y_coords))

    if transparent:
        # Create BGRA image with alpha channel
        bgra = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = mask  # Set alpha channel to mask
        crop = bgra[y1:y2, x1:x2]
    else:
        # Black background (for RAM++ input)
        masked = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
        crop = masked[y1:y2, x1:x2]

    return crop


def _encode_masked_png_base64(masked_crop: np.ndarray) -> str:
    """Encode a masked crop as PNG base64 while enforcing response payload limits."""
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


# ──────────────────────────────────────────────────────────────────────────────
# Step 4: RAM++ fine-grained semantic tagging
# ──────────────────────────────────────────────────────────────────────────────

def _tag_with_ram(crop_bgr: np.ndarray, fallback_label: str) -> str:
    """
    Runs RAM++ on a BGR crop and returns the most specific leaf-level tag.
    Falls back to fallback_label (the YOLO-World detector label) on any failure.
    """
    if crop_bgr.size == 0:
        return fallback_label
    try:
        import torch
        from PIL import Image as PILImage
        from ram import inference_ram_openset

        model, transform = _get_ram()
        if model is None:
            return fallback_label

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil = PILImage.fromarray(crop_rgb)
        tensor = transform(pil).unsqueeze(0)

        with torch.no_grad():
            tags_en = inference_ram_openset(tensor, model)

        specific = _pick_most_specific(tags_en)
        return specific if specific else fallback_label
    except Exception:
        return fallback_label


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_snap_learn(image_bytes: bytes, target_lang: str, max_objects: int) -> SnapLearnResponse:
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Could not decode image")

    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # ── Step 1: YOLO-World open-vocabulary detection ──────────────────────────
    raw_dets = _detect_yolo_world(image_rgb)

    # ── Step 2: Salient Object Ranking → top N ─────────────────────────────
    for det in raw_dets:
        det["saliency_score"] = _sor_score(det["bbox"], det["confidence"], w, h)
    raw_dets.sort(key=lambda d: d["saliency_score"], reverse=True)
    top_dets = raw_dets[:max_objects]

    # ── Steps 3 + 4: MobileSAM + RAM++ (per object) ────────────────────────
    detections: list[dict] = []
    for idx, det in enumerate(top_dets, start=1):
        x1 = max(0, det["bbox"][0])
        y1 = max(0, det["bbox"][1])
        x2 = min(w, det["bbox"][2])
        y2 = min(h, det["bbox"][3])
        bbox = [x1, y1, x2, y2]

        polygon = _segment_with_sam(image_rgb, bbox)

        # Extract masked crop (black background) for RAM++ tagging
        crop_bgr = _extract_masked_crop(image_bgr, polygon, transparent=False)
        canonical_tag = _tag_with_ram(crop_bgr, fallback_label=det["label"])

        # Generate transparent PNG for Frontend
        masked_crop_transparent = _extract_masked_crop(image_bgr, polygon, transparent=True)
        masked_b64 = _encode_masked_png_base64(masked_crop_transparent)

        detections.append({
            "object_id": f"obj_{idx}",
            "bbox": bbox,
            "polygon": polygon,
            "canonical_tag": canonical_tag,
            "confidence": det["confidence"],
            "masked_image_base64": masked_b64,
        })

    # ── Artifacts ──────────────────────────────────────────────────────────
    _save_artifacts(image_bgr, detections)

    # ── Step 5: Translation → flashcard ────────────────────────────────────
    objects: list[SnapLearnObject] = []
    for det in detections:
        tag = det["canonical_tag"]
        flashcard = build_flashcard(
            FlashcardRequest(
                user_id="local-test-user",
                object_id=det["object_id"],
                source_word=tag,
                source_lang="en",
                target_lang=target_lang,
                proficiency_level="A2",
            )
        )
        objects.append(
            SnapLearnObject(
                object_id=det["object_id"],
                bbox=det["bbox"],
                polygon=det["polygon"],
                canonical_tag=tag,
                confidence=det["confidence"],
                translated_word=flashcard.translated_word,
                ipa=flashcard.ipa,
                masked_image_base64=det["masked_image_base64"],
            )
        )

    return SnapLearnResponse(
        image_width=int(w),
        image_height=int(h),
        objects=objects,
        total_objects=len(objects),
    )
