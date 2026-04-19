import os
from io import BytesIO
from typing import Optional

import numpy as np
from PIL import Image

from app.core.config import settings
from app.utils.network_security import download_bytes_with_limit, parse_allowed_hosts

try:
    import cv2
    import onnxruntime as ort
except Exception:  # pragma: no cover
    cv2 = None
    ort = None


class LocalVisionONNX:
    """
    Local ONNX pipeline:
    1) YOLO-World ONNX -> detections
    2) MobileSAM ONNX -> mask/polygon for each detection
    3) RAM++ ONNX -> semantic tag for each crop

    If model files or dependencies are missing, caller should fallback to HTTP pipeline.
    """

    def __init__(self) -> None:
        if ort is None or cv2 is None:
            raise RuntimeError("onnxruntime and opencv-python are required for local vision backend")

        self.yolo_sess = ort.InferenceSession(settings.yolo_onnx_path, providers=["CPUExecutionProvider"])
        self.sam_sess = ort.InferenceSession(settings.mobilesam_onnx_path, providers=["CPUExecutionProvider"])
        self.rampp_sess = ort.InferenceSession(settings.rampp_onnx_path, providers=["CPUExecutionProvider"])
        self.labels = self._load_labels(settings.rampp_labels_path)

    def run(self, image_url: str, max_objects: int) -> list[dict]:
        image = self._fetch_image(image_url)
        h, w = image.shape[:2]

        detections = self._detect_yolo(image)
        detections = self._rank_salient(detections, max_objects, w, h)

        out: list[dict] = []
        for i, det in enumerate(detections, start=1):
            bbox = [int(x) for x in det["bbox"]]
            polygon = self._segment_polygon(image, bbox)
            crop = self._crop_bbox(image, bbox)
            canonical_tag = self._tag_rampp(crop)

            out.append(
                {
                    "object_id": f"obj_{i}",
                    "bbox": bbox,
                    "polygon": polygon,
                    "detector_label": str(det.get("label", "unknown")),
                    "canonical_tag": canonical_tag,
                    "confidence": float(det.get("confidence", 0.0)),
                    "source": "rampp",
                }
            )

        return out

    def _load_labels(self, path: str) -> list[str]:
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def _fetch_image(self, image_url: str) -> np.ndarray:
        image_bytes = download_bytes_with_limit(
            image_url,
            timeout_seconds=settings.vision_timeout_seconds,
            max_bytes=settings.max_remote_fetch_bytes,
            allowed_hosts=parse_allowed_hosts(settings.allowed_external_hosts),
            allow_private=settings.allow_private_network_urls,
            accepted_content_prefixes=("image/",),
        )
        pil = Image.open(BytesIO(image_bytes)).convert("RGB")
        return np.array(pil)

    def _detect_yolo(self, image: np.ndarray) -> list[dict]:
        # Generic YOLO ONNX preprocessing; adjust input shape if your exported model differs.
        inp_name = self.yolo_sess.get_inputs()[0].name
        _, _, in_h, in_w = self.yolo_sess.get_inputs()[0].shape
        resized = cv2.resize(image, (int(in_w), int(in_h)))
        x = resized.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, ...]
        outputs = self.yolo_sess.run(None, {inp_name: x})
        arr = np.array(outputs[0])

        # Supports common exported layouts: [1, N, 6] or [N, 6]
        if arr.ndim == 3:
            arr = arr[0]

        dets: list[dict] = []
        for row in arr:
            if len(row) < 6:
                continue
            x1, y1, x2, y2, conf, cls_idx = row[:6]
            if conf < 0.2:
                continue
            dets.append(
                {
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(conf),
                    "label": f"class_{int(cls_idx)}",
                }
            )
        return dets

    def _rank_salient(self, dets: list[dict], top_n: int, width: int, height: int) -> list[dict]:
        cx, cy = width / 2.0, height / 2.0
        max_dist = (cx**2 + cy**2) ** 0.5

        for det in dets:
            x1, y1, x2, y2 = det["bbox"]
            bx = (x1 + x2) / 2.0
            by = (y1 + y2) / 2.0
            dist = ((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5
            center_prox = max(0.0, 1.0 - (dist / max_dist))
            area = max(0.0, (x2 - x1) * (y2 - y1))
            area_ratio = min(1.0, area / float(width * height))

            score = 0.4 * det["confidence"] + 0.3 * center_prox + 0.3 * area_ratio
            det["saliency_score"] = score

        dets.sort(key=lambda d: d.get("saliency_score", 0.0), reverse=True)
        return dets[:top_n]

    def _segment_polygon(self, image: np.ndarray, bbox: list[int]) -> list[list[int]]:
        # MobileSAM ONNX call placeholder with robust fallback to bbox polygon.
        # Different exports have different prompt tensor names; adjust this method to your actual model.
        try:
            inp = self.sam_sess.get_inputs()
            if len(inp) == 0:
                raise RuntimeError("MobileSAM ONNX has no inputs")
            # No standard prompt signature across exports; return bbox polygon until mapped.
            raise RuntimeError("MobileSAM prompt mapping not configured")
        except Exception:
            x1, y1, x2, y2 = bbox
            return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

    def _crop_bbox(self, image: np.ndarray, bbox: list[int]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return image
        return image[y1:y2, x1:x2]

    def _tag_rampp(self, crop: np.ndarray) -> str:
        if crop.size == 0:
            return "unknown"

        inp_name = self.rampp_sess.get_inputs()[0].name
        _, _, in_h, in_w = self.rampp_sess.get_inputs()[0].shape
        resized = cv2.resize(crop, (int(in_w), int(in_h)))
        x = resized.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, ...]
        outputs = self.rampp_sess.run(None, {inp_name: x})
        logits = np.array(outputs[0]).reshape(-1)
        idx = int(np.argmax(logits)) if logits.size > 0 else -1

        if 0 <= idx < len(self.labels):
            return self.labels[idx]
        return f"label_{idx}" if idx >= 0 else "unknown"


def run_local_onnx(image_url: str, max_objects: int) -> Optional[list[dict]]:
    try:
        runner = LocalVisionONNX()
        return runner.run(image_url=image_url, max_objects=max_objects)
    except Exception:
        return None
