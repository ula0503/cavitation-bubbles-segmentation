"""YOLO-based segmentation module for bubble detection."""

from typing import List, Dict, Optional, Any
import numpy as np
from ultralytics import YOLO


class YoloSegmenter:
    """YOLO segmenter for bubble detection in video frames."""

    def __init__(self, model_path: str) -> None:
        """Initialize YOLO segmentation model.

        Args:
            model_path: Path to YOLO model weights file.
        """
        self.model = YOLO(model_path)

    def segment_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Process a single frame and return detections.

        Args:
            frame: Input image frame (BGR format).

        Returns:
            List of detection dictionaries with keys:
                - bbox: [x1, y1, x2, y2] bounding box coordinates
                - mask: Binary mask as np.ndarray (0 or 1 values)
                - class: Integer class ID (0: in focus, 1: out of focus)
                - confidence: Detection confidence score
        """
        # Run inference
        results = self.model(
            frame,
            conf=0.15,  # Confidence threshold
            iou=0.30,  # IoU threshold for NMS
            imgsz=1280,  # Inference image size
            retina_masks=True,  # High-quality masks
            agnostic_nms=False,  # Separate classes for NMS
        )

        # Get first result (single frame)
        result = results[0]
        detections = []

        # Extract bounding boxes, scores, and classes
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()  # shape: (n, 4)
            scores = result.boxes.conf.cpu().numpy()  # shape: (n,)
            classes = result.boxes.cls.cpu().numpy()  # shape: (n,)
        else:
            boxes, scores, classes = [], [], []

        # Extract masks if available
        if result.masks is not None:
            # Masks shape: (n, height, width) with values 0-1
            masks = result.masks.data.cpu().numpy()
        else:
            masks = [None] * len(boxes)

        # Build detection dictionaries
        for i, bbox in enumerate(boxes):
            mask = masks[i] if i < len(masks) else None

            # Convert mask to binary if exists
            binary_mask = None
            if mask is not None:
                binary_mask = (mask > 0.5).astype(np.uint8)

            detection = {
                "bbox": bbox.tolist(),
                "mask": binary_mask,
                "class": int(classes[i]),
                "confidence": float(scores[i]),
            }
            detections.append(detection)

        return detections
