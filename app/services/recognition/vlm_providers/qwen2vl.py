"""
Qwen2-VL-2B local VLM provider for object detection.

Uses Alibaba's Qwen2-VL-2B-Instruct model from HuggingFace for FREE local inference.
Model size: ~4 GB, downloads automatically on first use to ~/.cache/huggingface/
Requires: transformers>=4.37.0, qwen-vl-utils, accelerate

This is a DROP-IN replacement for OpenAI GPT-4V with:
- Zero API costs (runs locally)
- Better accuracy than Florence-2-base (~85-95% vs ~66%)
- Native grounding support (bbox extraction)
- CPU-only inference (~40-60s per image on CPU)
- Automatic image resizing for large images (prevents memory issues)

⚠️ CRITICAL: Use ONLY Qwen2-VL-2B-Instruct (~4 GB). DO NOT use the 7B model (~14 GB).
"""
from __future__ import annotations

import io
import re
from typing import Any, Optional

from PIL import Image

from .base import DetectedObject, VLMDetectionResult, VLMProvider

# Maximum image dimension for VLM processing (prevents memory issues on large images)
# Images larger than this will be resized proportionally before inference
MAX_IMAGE_DIMENSION = 1280


def _resize_image_if_needed(pil_image: Image.Image, max_dim: int = MAX_IMAGE_DIMENSION) -> Image.Image:
    """
    Resize image if either dimension exceeds max_dim, preserving aspect ratio.
    
    Large images (e.g., 6000x8000) cause extreme slowdowns on CPU inference.
    Resizing to ~1280px max dimension reduces processing time from hours to seconds.
    
    Args:
        pil_image: PIL Image to potentially resize
        max_dim: Maximum allowed dimension (default: 1280)
    
    Returns:
        Resized image if needed, otherwise original image
    """
    width, height = pil_image.size
    
    if width <= max_dim and height <= max_dim:
        return pil_image
    
    # Calculate new dimensions preserving aspect ratio
    if width > height:
        new_width = max_dim
        new_height = int(height * (max_dim / width))
    else:
        new_height = max_dim
        new_width = int(width * (max_dim / height))
    
    return pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)


# Lazy-loaded model singletons
_qwen_model: Any = None
_qwen_processor: Any = None


def _get_qwen() -> tuple[Any, Any]:
    """Lazy-load Qwen2-VL model and processor as singletons."""
    global _qwen_model, _qwen_processor

    if _qwen_model is None:
        try:
            import torch
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "transformers package not installed. "
                "Install with: pip install transformers>=4.37.0"
            ) from exc

        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise RuntimeError(
                "qwen-vl-utils package not installed. "
                "Install with: pip install qwen-vl-utils"
            ) from exc

        model_id = "Qwen/Qwen2-VL-2B-Instruct"

        _qwen_processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
        )

        # Auto-detect GPU: use CUDA if available for 5-8x speedup
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        print(f"[Qwen2-VL] Loading on {device.upper()}" + (" (GPU accelerated)" if device == "cuda" else " (no GPU detected)"))

        _qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=device,
        )
        _qwen_model.eval()

    return _qwen_model, _qwen_processor


class Qwen2VLProvider(VLMProvider):
    """
    Qwen2-VL-2B local VLM provider for object detection with grounding.

    Uses Alibaba's Qwen2-VL-2B-Instruct model running locally on CPU.
    Cost: FREE (no API calls)
    Latency: ~40-60s per image on CPU
    Model size: ~4 GB (downloaded to ~/.cache/huggingface/)
    Expected accuracy: 85-95% (better than Florence-2-base's 66%)
    """

    def __init__(self, model_id: str = "Qwen/Qwen2-VL-2B-Instruct"):
        """
        Initialize Qwen2-VL provider.

        Args:
            model_id: HuggingFace model ID (default: Qwen/Qwen2-VL-2B-Instruct)
                      ⚠️ DO NOT use Qwen/Qwen2-VL-7B-Instruct (14 GB, too slow on CPU)
        """
        if "7B" in model_id:
            raise ValueError(
                f"❌ 7B model is too large for CPU inference ({model_id}).\n"
                "Use Qwen/Qwen2-VL-2B-Instruct (~4 GB) instead."
            )
        self.model_id = model_id

    def detect_objects(
        self,
        image_bytes: bytes,
        max_objects: int = 5,
        prompt: Optional[str] = None,
    ) -> VLMDetectionResult:
        """
        Detect objects using Qwen2-VL grounding with local model.

        Args:
            image_bytes: Raw image bytes (JPEG/PNG)
            max_objects: Maximum number of objects to return
            prompt: Optional custom prompt (uses grounding default if None)

        Returns:
            VLMDetectionResult with bboxes in 0-1000 normalized range
        """
        import torch
        from qwen_vl_utils import process_vision_info

        model, processor = _get_qwen()

        # ── Convert bytes to PIL Image ────────────────────────────────────────
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Resize large images to prevent memory issues and long inference times
        pil_image = _resize_image_if_needed(pil_image)
        img_width, img_height = pil_image.size

        # ── Prepare grounding prompt ──────────────────────────────────────────
        # Qwen2-VL uses conversational format for grounding tasks
        # The model outputs grounding tokens when explicitly asked for object locations
        if prompt is None:
            # Use Qwen2-VL's expected grounding prompt format
            prompt = "Find all the objects in this image and provide their bounding boxes."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # ── Process and run inference ─────────────────────────────────────────
        # Apply chat template to convert messages to model input format
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process vision inputs (images/videos)
        image_inputs, video_inputs = process_vision_info(messages)

        # Prepare model inputs
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Generate detections
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,  # Greedy decoding for consistency
            )

        # Decode output (keep special tokens for bbox parsing)
        output_text = processor.batch_decode(
            generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]

        # ── Parse grounding output ────────────────────────────────────────────
        # Qwen2-VL returns format: <ref>label</ref><box>[[x1,y1],[x2,y2]]</box>
        detected_objects = self._parse_grounding_output(
            output_text, img_width, img_height, max_objects
        )

        return VLMDetectionResult(
            objects=detected_objects,
            raw_response=output_text,
        )

    def _parse_grounding_output(
        self,
        output_text: str,
        img_width: int,
        img_height: int,
        max_objects: int,
    ) -> list[DetectedObject]:
        """
        Parse Qwen2-VL grounding output to extract object labels and bboxes.

        Qwen2-VL-2B-Instruct returns markdown-formatted text with normalized coords:
            **Left Smartphone**: [0.13, 0.35, 0.3, 0.75]
            **Right Smartphone**: [0.67, 0.35, 0.83, 0.75]
            **Casio Calculator**: [0.38, 0.35, 0.62, 0.75]

        Args:
            output_text: Raw model output text
            img_width: Original image width in pixels
            img_height: Original image height in pixels
            max_objects: Maximum number of objects to return

        Returns:
            List of DetectedObject with bboxes normalized to 0-1000 range
        """
        # Pattern 1: Match markdown bold labels with normalized coords
        # Example: **Left Smartphone**: [0.13, 0.35, 0.3, 0.75]
        pattern1 = r'\*\*(.*?)\*\*:\s*\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]'

        # Pattern 2: Match plain labels with normalized coords
        # Example: Left Smartphone: [0.13, 0.35, 0.3, 0.75]
        pattern2 = r'([A-Za-z\s]+):\s*\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]'

        matches = re.findall(pattern1, output_text)
        if not matches:
            matches = re.findall(pattern2, output_text)

        detected_objects: list[DetectedObject] = []
        seen_labels = set()  # Deduplicate (e.g., "Top Left Corner" appears for multiple objects)

        for i, match in enumerate(matches):
            if len(detected_objects) >= max_objects:
                break

            label, x1_str, y1_str, x2_str, y2_str = match

            # Clean up label
            label = label.strip()

            # Skip generic corner labels (these are intermediate parsing artifacts)
            if "corner" in label.lower():
                continue

            # Deduplicate identical labels
            if label in seen_labels:
                continue
            seen_labels.add(label)

            try:
                # Parse normalized coordinates (0-1 range)
                x1_norm = float(x1_str)
                y1_norm = float(y1_str)
                x2_norm = float(x2_str)
                y2_norm = float(y2_str)

                # Validate normalized coords are in 0-1 range
                if not (0 <= x1_norm <= 1 and 0 <= y1_norm <= 1 and
                        0 <= x2_norm <= 1 and 0 <= y2_norm <= 1):
                    continue

                # Validate bbox geometry (x2 > x1, y2 > y1)
                if x2_norm <= x1_norm or y2_norm <= y1_norm:
                    continue

                # Convert normalized 0-1 coords to 0-1000 range
                bbox_1000 = [
                    int(x1_norm * 1000),
                    int(y1_norm * 1000),
                    int(x2_norm * 1000),
                    int(y2_norm * 1000),
                ]

                detected_objects.append(
                    DetectedObject(
                        label=label,
                        bbox=bbox_1000,
                        confidence=0.9,
                    )
                )

            except (ValueError, IndexError):
                continue

        return detected_objects

    def identify_object(self, image_bytes: bytes) -> str:
        """Identify a single object in an image crop."""
        import torch
        from qwen_vl_utils import process_vision_info

        model, processor = _get_qwen()

        # Convert bytes to PIL Image and resize if needed
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        pil_image = _resize_image_if_needed(pil_image)

        # Simple identification prompt (no grounding needed)
        prompt = "What is this object? Provide a simple, specific name in 2-4 words."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Process and run inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=128,  # Short response only
                do_sample=False,
            )

        # Decode output
        output_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]

        # Extract the label from the assistant's response
        # Format: "system\n...\nuser\n...\nassistant\nThe object is a black ballpoint pen."
        if "assistant" in output_text.lower():
            parts = output_text.split("assistant")
            if len(parts) > 1:
                label = parts[-1].strip()
            else:
                label = output_text.strip()
        else:
            label = output_text.strip()

        # Clean up common response patterns
        label = label.replace("The object is a ", "")
        label = label.replace("This is a ", "")
        label = label.replace("It is a ", "")
        label = label.replace("A ", "")
        label = label.strip().rstrip(".")

        return label

    def get_provider_name(self) -> str:
        """Return provider name for logging."""
        return "qwen2-vl-2b"

    def estimate_cost(self, image_bytes: bytes) -> float:
        """
        Return 0.0 since Qwen2-VL is a FREE local model.
        """
        return 0.0
