"""
Base interface for VLM providers.
All VLM implementations (OpenAI, Anthropic, Google, local models) must inherit from this.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class DetectedObject:
    """Single object detected by VLM."""
    label: str  # Natural language description (e.g., "beach umbrella", "calculator")
    bbox: list[int]  # [x_min, y_min, x_max, y_max] in pixels
    confidence: float  # 0.0 to 1.0


@dataclass
class VLMDetectionResult:
    """Complete detection result from VLM."""
    objects: list[DetectedObject]
    raw_response: Optional[str] = None  # For debugging


class VLMProvider(ABC):
    """
    Abstract base class for Vision-Language Model providers.

    Each provider implements detection and returns DetectedObject instances
    with bounding boxes and natural language labels.
    """

    @abstractmethod
    def detect_objects(
        self,
        image_bytes: bytes,
        max_objects: int = 5,
        prompt: Optional[str] = None,
    ) -> VLMDetectionResult:
        """
        Detect objects in an image using the VLM.

        Args:
            image_bytes: Raw image bytes (JPEG/PNG)
            max_objects: Maximum number of objects to detect
            prompt: Optional custom prompt (uses default if None)

        Returns:
            VLMDetectionResult with detected objects and bounding boxes

        Raises:
            RuntimeError: If API call fails
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return provider name for logging (e.g., 'openai-gpt4v')."""
        pass

    @abstractmethod
    def estimate_cost(self, image_bytes: bytes) -> float:
        """
        Estimate cost in USD for processing this image.

        Returns:
            Estimated cost in dollars (e.g., 0.01 for $0.01)
        """
        pass
