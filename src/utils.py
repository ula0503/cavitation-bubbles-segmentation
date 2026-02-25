import numpy as np
import math
import cv2


def compute_centroid(bbox):
    """
    Вычисляет центр bounding box в формате [x1, y1, x2, y2].
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def euclidean_distance(p1, p2):
    """
    Вычисляет евклидово расстояние между двумя точками.
    """
    if p1 is None or p2 is None:
        return float("inf")
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def draw_mask(frame, mask, color, border_width=2, line_type=cv2.LINE_AA):
    """
    Обводка с дополнительными настройками.
    line_type=cv2.LINE_AA - сглаженные линии
    """
    if mask is None or mask.size == 0:
        return frame

    mask = mask.astype(np.uint8)
    if mask.shape[:2] != frame.shape[:2]:
        mask = cv2.resize(
            mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST
        )

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, color, border_width, lineType=line_type)

    return frame
