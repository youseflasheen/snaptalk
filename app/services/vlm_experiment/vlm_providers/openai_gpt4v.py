"""
OpenAI GPT-4 Vision provider for object detection.

Requires: pip install openai
API key: Set OPENAI_API_KEY environment variable or pass to constructor
"""
from __future__ import annotations

import base64
import json
import os
from typing import Optional

from .base import DetectedObject, VLMDetectionResult, VLMProvider


class OpenAIGPT4VProvider(VLMProvider):
    """
    GPT-4 Vision API provider for object detection.

    Uses GPT-4V with function calling to get structured detection results with
    bounding boxes in a reliable format.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model to use (gpt-4o, gpt-4o-mini, gpt-4-turbo, etc.)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self.model = model

    def detect_objects(
        self,
        image_bytes: bytes,
        max_objects: int = 5,
        prompt: Optional[str] = None,
    ) -> VLMDetectionResult:
        """Detect objects using GPT-4 Vision API."""
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai package not installed. Install with: pip install openai"
            ) from exc

        client = OpenAI(api_key=self.api_key)

        # Encode image to base64
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')

        # Default prompt optimized for language learning apps
        if prompt is None:
            prompt = f"""You are an object detection assistant for a language learning app.

Identify the {max_objects} most prominent everyday objects in this image that would be useful for vocabulary learning.

For each object, provide:
1. A clear, simple name (preferably 1-2 words, suitable for flashcards)
2. Bounding box coordinates as [x_min, y_min, x_max, y_max] in pixels, normalized to 0-1000 range
3. Confidence score (0.0 to 1.0)

Focus on concrete, tangible objects (avoid abstract concepts like "scene" or "background").
Prefer common everyday items that language learners would encounter."""

        # Function schema for structured output
        functions = [
            {
                "name": "detect_objects",
                "description": "Return detected objects with bounding boxes",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "objects": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "label": {
                                        "type": "string",
                                        "description": "Object name (1-2 words, e.g., 'beach umbrella', 'calculator')",
                                    },
                                    "bbox": {
                                        "type": "array",
                                        "items": {"type": "number"},
                                        "minItems": 4,
                                        "maxItems": 4,
                                        "description": "Bounding box [x_min, y_min, x_max, y_max] in 0-1000 range",
                                    },
                                    "confidence": {
                                        "type": "number",
                                        "minimum": 0.0,
                                        "maximum": 1.0,
                                        "description": "Confidence score",
                                    },
                                },
                                "required": ["label", "bbox", "confidence"],
                            },
                        }
                    },
                    "required": ["objects"],
                },
            }
        ]

        # Make API call
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}",
                                    "detail": "high",
                                },
                            },
                        ],
                    }
                ],
                functions=functions,
                function_call={"name": "detect_objects"},
                max_tokens=1000,
            )

            # Parse function call result
            function_call = response.choices[0].message.function_call
            if not function_call:
                raise RuntimeError("GPT-4V did not return function call")

            result = json.loads(function_call.arguments)
            objects_data = result.get("objects", [])

            # Convert to DetectedObject instances
            detected_objects = []
            for obj in objects_data[:max_objects]:
                # Normalize bbox from 0-1000 range to actual pixel coordinates
                # (will be done by caller based on actual image dimensions)
                detected_objects.append(
                    DetectedObject(
                        label=obj["label"],
                        bbox=[int(x) for x in obj["bbox"]],
                        confidence=float(obj["confidence"]),
                    )
                )

            return VLMDetectionResult(
                objects=detected_objects,
                raw_response=function_call.arguments,
            )

        except Exception as exc:
            raise RuntimeError(f"OpenAI API call failed: {exc}") from exc

    def get_provider_name(self) -> str:
        return f"openai-{self.model}"

    def estimate_cost(self, image_bytes: bytes) -> float:
        """
        Estimate cost based on GPT-4V pricing (as of 2024).

        Pricing (approximate):
        - gpt-4o: $0.005 per image (high detail)
        - gpt-4o-mini: $0.001 per image
        - gpt-4-turbo: $0.01 per image (high detail)
        """
        # Size-based pricing (high detail)
        size_mb = len(image_bytes) / (1024 * 1024)

        cost_map = {
            "gpt-4o": 0.005,
            "gpt-4o-mini": 0.001,
            "gpt-4-turbo": 0.01,
            "gpt-4-vision-preview": 0.01,
        }

        base_cost = cost_map.get(self.model, 0.005)  # Default to gpt-4o pricing
        return base_cost * (1 + size_mb * 0.1)  # Slight adjustment for large images
