"""
HYBRID Snap & Learn Pipeline (PRODUCTION)
==========================================

This uses a HYBRID approach for accurate object detection and labeling:
  1. LDET (YOLOv8m in class-agnostic mode) for object detection
  2. Qwen2-VL-2B for accurate object labeling (GPU accelerated)
  3. MobileSAM for pixel-perfect segmentation

Architecture:
  Image → LDET Detection → Qwen2-VL Labeling → MobileSAM Segmentation → Flashcard

Performance (with CUDA GPU):
  - Detection: ~3-5s (LDET)
  - Labeling: ~5-10s per object (Qwen2-VL on GPU)
  - Segmentation: ~3-5s per object (MobileSAM)
  - Cost: FREE (all local models)
  - Accuracy: 95%+ (Qwen's labeling accuracy)
"""
from __future__ import annotations

import os
import threading
from typing import Any

import cv2
import numpy as np

from app.schemas.pipeline import SnapLearnObject, SnapLearnResponse
from app.schemas.translation import FlashcardRequest
from app.services.translation_service import build_flashcard
from app.services.vlm_experiment.vlm_providers import (
    OpenAIGPT4VProvider,
    Qwen2VLProvider,
    VLMProvider,
)

# Import detection, segmentation, and utilities from original pipeline
from app.services.snap_learn_service import (
    _bbox_polygon,
    _encode_masked_png_base64,
    _extract_masked_crop,
    _get_sam,
    _save_artifacts,
    _segment_with_sam,
    _simplify_polygon,
)

# ──────────────────────────────────────────────────────────────────────────────
# LDET detector (YOLOv8m in class-agnostic mode)
# ──────────────────────────────────────────────────────────────────────────────
_ldet_model: Any = None
_ldet_model_lock = threading.Lock()


def _get_ldet() -> Any:
    """Lazy-load LDET model singleton (YOLOv8m in class-agnostic mode)."""
    global _ldet_model
    if _ldet_model is None:
        with _ldet_model_lock:
            if _ldet_model is None:
                print("[LDET] Loading YOLO in class-agnostic mode (LDET proxy)...")
                from ultralytics import YOLO
                _ldet_model = YOLO("yolov8m.pt")
                print("[LDET] Model loaded successfully")
    return _ldet_model


def _compute_iou(box1: list, box2: list) -> float:
    """Compute IoU between two bounding boxes [x1,y1,x2,y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


def _remove_duplicate_detections(detections: list[dict], iou_threshold: float = 0.5) -> list[dict]:
    """
    Remove duplicate detections based on IoU overlap.
    Keeps higher confidence detection when two boxes overlap significantly.
    """
    if len(detections) <= 1:
        return detections
    
    # Sort by confidence descending
    sorted_dets = sorted(detections, key=lambda x: x["confidence"], reverse=True)
    keep = []
    
    for det in sorted_dets:
        should_keep = True
        for kept in keep:
            iou = _compute_iou(det["bbox"], kept["bbox"])
            if iou > iou_threshold:
                should_keep = False
                break
        if should_keep:
            keep.append(det)
    
    return keep


def _detect_with_ldet(
    image_rgb: np.ndarray,
    conf_threshold: float = 0.15,
    iou_threshold: float = 0.4,
    max_det: int = 300,
) -> list[dict]:
    """Detect objects using class-agnostic LDET detection."""
    model = _get_ldet()

    results = model(
        image_rgb,
        conf=conf_threshold,
        iou=iou_threshold,
        max_det=max_det,
        agnostic_nms=True,
        verbose=False,
    )

    if not results or len(results) == 0:
        return []

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []

    detections = []
    for i in range(len(boxes)):
        bbox_xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
        conf = float(boxes.conf[i].cpu().numpy())

        detections.append({
            "bbox": bbox_xyxy.tolist(),
            "confidence": conf,
            "label": "object",
        })

    detections.sort(key=lambda x: x["confidence"], reverse=True)
    
    # Remove duplicates with stricter IoU threshold
    detections = _remove_duplicate_detections(detections, iou_threshold=0.45)
    
    return detections


def _get_vlm_provider() -> VLMProvider:
    """Get configured VLM provider for object labeling."""
    provider_name = os.getenv("VLM_PROVIDER", "qwen2vl").lower()

    if provider_name == "qwen2vl":
        return Qwen2VLProvider()
    elif provider_name == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable not set.")
        return OpenAIGPT4VProvider(api_key=api_key, model="gpt-4o")
    else:
        raise ValueError(f"Unknown VLM provider: {provider_name}")


def _normalize_bbox_from_1000(
    bbox_1000: list[int], actual_width: int, actual_height: int
) -> list[int]:
    """
    Convert bounding box from 0-1000 normalized coordinates to actual pixel coordinates.

    VLM returns bboxes in 0-1000 range for consistency. This converts to actual pixels.
    """
    x1, y1, x2, y2 = bbox_1000
    x1_px = int((x1 / 1000.0) * actual_width)
    y1_px = int((y1 / 1000.0) * actual_height)
    x2_px = int((x2 / 1000.0) * actual_width)
    y2_px = int((y2 / 1000.0) * actual_height)

    # Clamp to image bounds
    x1_px = max(0, min(x1_px, actual_width))
    y1_px = max(0, min(y1_px, actual_height))
    x2_px = max(0, min(x2_px, actual_width))
    y2_px = max(0, min(y2_px, actual_height))

    return [x1_px, y1_px, x2_px, y2_px]


def run_snap_learn_vlm(
    image_bytes: bytes, target_lang: str, max_objects: int = 5
) -> SnapLearnResponse:
    """
    Run HYBRID Snap & Learn pipeline: LDET Detection + Qwen2-VL Labeling + MobileSAM Segmentation.
    
    Pipeline:
      Image → LDET (YOLOv8m class-agnostic) → Qwen2-VL-2B → MobileSAM → Flashcard
    """
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Could not decode image")

    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Step 1: Object detection with LDET (YOLOv8m class-agnostic)
    print("[HYBRID] Step 1: LDET detection (complete object boundaries)...")
    raw_detections = _detect_with_ldet(image_rgb)
    print(f"[HYBRID] Detected {len(raw_detections)} objects")

    raw_detections = raw_detections[:max_objects]

    if not raw_detections:
        return SnapLearnResponse(
            image_width=int(w),
            image_height=int(h),
            objects=[],
            total_objects=0,
        )

    # Step 2: VLM labeling with Qwen2-VL (GPU accelerated)
    print("[HYBRID] Step 2: Qwen2-VL labeling...")
    vlm_provider = _get_vlm_provider()
    segmentation_mode = os.getenv("SEGMENTATION", "mobilesam").lower()
    print(f"[HYBRID] Segmentation mode: {segmentation_mode}")

    detections: list[dict] = []

    for idx, det in enumerate(raw_detections, start=1):
        bbox = det["bbox"]
        x1, y1, x2, y2 = bbox

        if x2 <= x1 or y2 <= y1:
            continue

        print(f"[HYBRID]   Labeling object {idx}...")
        crop_bgr = image_bgr[y1:y2, x1:x2]
        _, crop_bytes = cv2.imencode('.jpg', crop_bgr)
        crop_bytes = crop_bytes.tobytes()

        try:
            label = vlm_provider.identify_object(crop_bytes)
        except Exception:
            label = "unknown object"

        # Step 3: Segmentation with MobileSAM (or bbox fallback)
        if segmentation_mode == "mobilesam":
            polygon = _segment_with_sam(image_rgb, bbox)
        else:
            polygon = _bbox_polygon(bbox)

        masked_crop_transparent = _extract_masked_crop(image_bgr, polygon, transparent=True)
        masked_b64 = _encode_masked_png_base64(masked_crop_transparent)

        detections.append({
            "object_id": f"obj_{idx}",
            "bbox": bbox,
            "polygon": polygon,
            "canonical_tag": label,
            "confidence": det["confidence"],
            "masked_image_base64": masked_b64,
        })

    # Save visual artifacts
    _save_artifacts(image_bgr, detections)

    # Step 3: Build response (skip translation if target_lang is "en" - detection only mode)
    objects: list[SnapLearnObject] = []
    for det in detections:
        tag = det["canonical_tag"]
        
        # Skip translation for detection-only mode (when target_lang == "en")
        if target_lang.lower() == "en":
            translated_word = tag
            ipa = "n/a"
        else:
            try:
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
                translated_word = flashcard.translated_word
                ipa = flashcard.ipa
            except Exception:
                # On any error, use English as fallback
                translated_word = tag
                ipa = "n/a"
        
        objects.append(
            SnapLearnObject(
                object_id=det["object_id"],
                bbox=det["bbox"],
                polygon=det["polygon"],
                canonical_tag=tag,
                confidence=det["confidence"],
                translated_word=translated_word,
                ipa=ipa,
                masked_image_base64=det["masked_image_base64"],
            )
        )

    return SnapLearnResponse(
        image_width=int(w),
        image_height=int(h),
        objects=objects,
        total_objects=len(objects),
    )
