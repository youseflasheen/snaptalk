"""
VLM-based Snap & Learn pipeline (EXPERIMENTAL)
================================================
This is an isolated experimental implementation that replaces YOLO-World + RAM++
with a Vision-Language Model for object detection and tagging.

Architecture:
  Step 1: VLM API call → object detection + natural language descriptions + bounding boxes
  Step 2: MobileSAM segmentation (reused from original pipeline)
  Step 3: Transparent PNG generation (reused from original pipeline)
  Step 4: Translation (reused from original pipeline)

The original pipeline in app/services/snap_learn_service.py is preserved unchanged.
"""
