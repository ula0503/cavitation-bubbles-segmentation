"""Utility functions for bubble detection and tracking."""

import math
from typing import Tuple, Optional, List, Any
import numpy as np
import cv2


def compute_centroid(bbox: List[float]) -> Tuple[float, float]:
    """Compute centroid of bounding box.

    Args:
        bbox: Bounding box in [x1, y1, x2, y2] format.

    Returns:
        Tuple of (x, y) centroid coordinates.
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def euclidean_distance(
    p1: Optional[Tuple[float, float]], p2: Optional[Tuple[float, float]]
) -> float:
    """Compute Euclidean distance between two points.

    Args:
        p1: First point as (x, y) tuple.
        p2: Second point as (x, y) tuple.

    Returns:
        Euclidean distance. Returns infinity if either point is None.
    """
    if p1 is None or p2 is None:
        return float("inf")

    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def draw_mask(
    frame: np.ndarray,
    mask: Optional[np.ndarray],
    color: Tuple[int, int, int],
    border_width: int = 2,
    line_type: int = cv2.LINE_AA,
) -> np.ndarray:
    """Draw mask outline on frame.

    Args:
        frame: Input image frame (BGR format).
        mask: Binary mask (0 or 255 values). Can be None or empty.
        color: BGR color tuple for outline.
        border_width: Width of outline border in pixels.
        line_type: OpenCV line type (default: cv2.LINE_AA for anti-aliased).

    Returns:
        Frame with mask outline drawn.
    """
    if mask is None or mask.size == 0:
        return frame

    # Ensure mask is uint8
    mask = mask.astype(np.uint8)

    # Resize mask if dimensions don't match frame
    if mask.shape[:2] != frame.shape[:2]:
        mask = cv2.resize(
            mask,
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    # Find contours and draw them
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(
        frame,
        contours,
        -1,  # Draw all contours
        color,
        border_width,
        lineType=line_type,
    )

    return frame
