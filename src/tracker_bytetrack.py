"""ByteTrack tracker implementation with Kalman filter for bubble tracking."""

import math
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(bbox1: List[float], bbox2: List[float]) -> float:
    """Calculate Intersection over Union for two bounding boxes.

    Args:
        bbox1: First bounding box in [x1, y1, x2, y2] format.
        bbox2: Second bounding box in [x1, y1, x2, y2] format.

    Returns:
        IoU value between 0 and 1.
    """
    # Calculate intersection
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate areas
    area1 = max(0, bbox1[2] - bbox1[0]) * max(0, bbox1[3] - bbox1[1])
    area2 = max(0, bbox2[2] - bbox2[0]) * max(0, bbox2[3] - bbox2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


class KalmanBoxTracker:
    """Kalman filter based tracker for individual objects.

    State: [x, y, s, r, vx, vy, vs]
        x, y - center coordinates
        s - scale (area)
        r - aspect ratio
        vx, vy, vs - corresponding velocities
    """

    count: int = 0  # Class variable for unique ID assignment

    def __init__(
        self,
        bbox: List[float],
        frame_idx: int,
        timestamp: float,
        detection: Optional[Dict] = None,
    ) -> None:
        """Initialize tracker with first detection.

        Args:
            bbox: Bounding box in [x1, y1, x2, y2] format.
            frame_idx: Current frame index.
            timestamp: Current timestamp in seconds.
            detection: Optional detection data containing mask and confidence.
        """
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        # Convert bbox to measurement: [x, y, s, r]
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        x = x1 + w / 2.0
        y = y1 + h / 2.0
        s = w * h
        r = w / (h + 1e-6)

        # Initialize state: [x, y, s, r, vx, vy, vs]
        self.state: np.ndarray = np.array([x, y, s, r, 0, 0, 0], dtype=np.float32)

        # Initial covariance
        self.P: np.ndarray = np.diag([10, 10, 100, 10, 1000, 1000, 1000]).astype(
            np.float32
        )

        # Default time step
        dt = 1.0
        self._update_transition_matrix(dt)

        # Measurement matrix H (measure only [x, y, s, r])
        self.H: np.ndarray = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ],
            dtype=np.float32,
        )

        # Process noise matrix Q
        self.Q: np.ndarray = np.diag([0.1, 0.1, 0.1, 0.001, 50, 50, 50]).astype(
            np.float32
        )

        # Measurement noise matrix R
        self.R: np.ndarray = np.diag([0.5, 0.5, 10, 0.01]).astype(np.float32)

        self.frame_idx: int = frame_idx
        self.timestamp: float = timestamp
        self.time_since_update: int = 0
        self.history: List[List[float]] = [bbox]  # History contains bboxes
        self.detection: Optional[Dict] = detection
        self.confidence: float = detection.get("confidence", 0.0) if detection else 0.0

    def _update_transition_matrix(self, dt: float) -> None:
        """Update state transition matrix F with new time step.

        Args:
            dt: Time step in seconds.
        """
        self.F = np.array(
            [
                [1, 0, 0, 0, dt, 0, 0],
                [0, 1, 0, 0, 0, dt, 0],
                [0, 0, 1, 0, 0, 0, dt],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

    def predict(self, dt: Optional[float] = None) -> List[float]:
        """Predict new state.

        Args:
            dt: Time step (defaults to 1.0).

        Returns:
            Predicted bbox in [x1, y1, x2, y2] format.
        """
        if dt is None:
            dt = 1.0

        self._update_transition_matrix(dt)

        self.state = np.dot(self.F, self.state)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        self.time_since_update += 1

        predicted_bbox = self.get_state()
        self.history.append(predicted_bbox)

        return predicted_bbox

    def update(
        self,
        bbox: List[float],
        frame_idx: int,
        timestamp: float,
        detection: Optional[Dict] = None,
    ) -> None:
        """Update state with new detection.

        Args:
            bbox: Bounding box in [x1, y1, x2, y2] format.
            frame_idx: Current frame index.
            timestamp: Current timestamp in seconds.
            detection: Detection data (always saved, even with low confidence).
        """
        # Calculate dt from timestamps
        dt = timestamp - self.timestamp if self.timestamp is not None else 1.0
        if dt <= 0:
            dt = 1.0

        self._update_transition_matrix(dt)

        # Convert bbox to measurement: [x, y, s, r]
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        x = x1 + w / 2.0
        y = y1 + h / 2.0
        s = w * h
        r = w / (h + 1e-6)
        z = np.array([x, y, s, r], dtype=np.float32)

        # Kalman update steps
        y_meas = z - np.dot(self.H, self.state)  # Innovation
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R  # Innovation covariance
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # Kalman gain

        self.state = self.state + np.dot(K, y_meas)  # State update
        I = np.eye(self.F.shape[0], dtype=np.float32)
        self.P = np.dot((I - np.dot(K, self.H)), self.P)  # Covariance update

        self.time_since_update = 0
        self.frame_idx = frame_idx
        self.timestamp = timestamp
        self.history.append(bbox)

        # Always save detection, even with low confidence
        self.detection = detection
        if detection is not None:
            self.confidence = detection.get("confidence", self.confidence)

    def get_state(self) -> List[float]:
        """Return current bbox in [x1, y1, x2, y2] format.

        Returns:
            Bounding box coordinates.
        """
        x, y, s, r = self.state[0:4]
        s = max(s, 1e-6)
        r = max(r, 1e-6)
        w = math.sqrt(s * r)
        h = s / (w + 1e-6)
        x1 = x - w / 2.0
        y1 = y - h / 2.0
        x2 = x + w / 2.0
        y2 = y + h / 2.0
        return [x1, y1, x2, y2]


class ByteTracker:
    """ByteTrack algorithm for multi-object tracking.

    Associates detections to tracks using IoU and distance thresholds.
    """

    def __init__(
        self,
        high_thresh: float = 0.6,
        low_thresh: float = 0.1,
        max_time_lost: int = 30,
        iou_threshold: float = 0.2,
        distance_threshold: float = 50,
    ) -> None:
        """Initialize ByteTracker.

        Args:
            high_thresh: Confidence threshold for high-quality detections.
            low_thresh: Confidence threshold for low-quality detections.
            max_time_lost: Max frames to keep track without updates.
            iou_threshold: IoU threshold for detection-track association.
            distance_threshold: Max Euclidean distance for association.
        """
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.max_time_lost = max_time_lost
        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold
        self.trackers: List[KalmanBoxTracker] = []
        self.finished_tracks: List[KalmanBoxTracker] = []

    def update(
        self, detections: List[Dict], frame_idx: int, timestamp: float
    ) -> Dict[int, KalmanBoxTracker]:
        """Update tracker with new detections.

        Args:
            detections: List of detection dicts with 'bbox', 'confidence', 'mask'.
            frame_idx: Current frame index.
            timestamp: Current timestamp in seconds.

        Returns:
            Dictionary of active tracks: {tracker_id: tracker_instance}.
        """
        # Split detections into high and low confidence
        high_detections = [
            det for det in detections if det["confidence"] >= self.high_thresh
        ]
        low_detections = [
            det
            for det in detections
            if self.low_thresh <= det["confidence"] < self.high_thresh
        ]

        high_boxes = (
            np.array([det["bbox"] for det in high_detections])
            if high_detections
            else np.empty((0, 4))
        )
        low_boxes = (
            np.array([det["bbox"] for det in low_detections])
            if low_detections
            else np.empty((0, 4))
        )

        # Predict new positions for all trackers
        for tracker in self.trackers:
            dt = timestamp - tracker.timestamp if tracker.timestamp is not None else 1.0
            if dt <= 0:
                dt = 1.0
            tracker.predict(dt)

        predicted_boxes = (
            np.array([tracker.get_state() for tracker in self.trackers])
            if self.trackers
            else np.empty((0, 4))
        )

        # Step 1: Associate high confidence detections with existing tracks
        matches, unmatched_trackers, unmatched_detections = (
            self._associate_detections_to_trackers(predicted_boxes, high_boxes)
        )

        # Update matched tracks
        for tracker_idx, detection_idx in matches:
            self.trackers[tracker_idx].update(
                high_boxes[detection_idx],
                frame_idx,
                timestamp,
                detection=high_detections[detection_idx],
            )

        # Step 2: Try to associate low confidence detections with remaining tracks
        if len(unmatched_trackers) > 0 and low_boxes.shape[0] > 0:
            unmatched_predicted = predicted_boxes[unmatched_trackers]
            matches_low, unmatched_trackers_final, _ = (
                self._associate_detections_to_trackers(unmatched_predicted, low_boxes)
            )

            for local_tracker_idx, detection_idx in matches_low:
                global_tracker_idx = unmatched_trackers[local_tracker_idx]
                self.trackers[global_tracker_idx].update(
                    low_boxes[detection_idx],
                    frame_idx,
                    timestamp,
                    detection=low_detections[detection_idx],
                )

            unmatched_trackers = [
                unmatched_trackers[i] for i in unmatched_trackers_final
            ]

        # Step 3: Increase lost count for unmatched tracks
        for idx in unmatched_trackers:
            self.trackers[idx].time_since_update += 1

        # Step 4: Create new tracks for unmatched high confidence detections
        for detection_idx in unmatched_detections:
            det = high_detections[detection_idx]
            new_tracker = KalmanBoxTracker(
                det["bbox"], frame_idx, timestamp, detection=det
            )
            self.trackers.append(new_tracker)

        # Step 5: Remove tracks that have been lost for too long
        active_trackers = []
        for tracker in self.trackers:
            if tracker.time_since_update > self.max_time_lost:
                self.finished_tracks.append(tracker)
            else:
                active_trackers.append(tracker)
        self.trackers = active_trackers

        # Return active tracks (updated within last frame)
        active = {
            tracker.id: tracker
            for tracker in self.trackers
            if tracker.time_since_update <= 1
        }
        return active

    def _associate_detections_to_trackers(
        self,
        trackers_boxes: np.ndarray,
        detections_boxes: np.ndarray,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Associate detections to tracks using IoU and distance.

        Args:
            trackers_boxes: Array of tracker bounding boxes.
            detections_boxes: Array of detection bounding boxes.

        Returns:
            Tuple containing:
                - matches: List of (tracker_idx, detection_idx) pairs
                - unmatched_trackers: List of tracker indices without matches
                - unmatched_detections: List of detection indices without matches
        """
        if trackers_boxes.shape[0] == 0 or detections_boxes.shape[0] == 0:
            return (
                [],
                list(range(trackers_boxes.shape[0])),
                list(range(detections_boxes.shape[0])),
            )

        # Compute IoU matrix
        iou_matrix = np.zeros(
            (trackers_boxes.shape[0], detections_boxes.shape[0]), dtype=np.float32
        )
        for t, tb in enumerate(trackers_boxes):
            for d, db in enumerate(detections_boxes):
                iou_matrix[t, d] = iou(tb.tolist(), db.tolist())

        # Hungarian algorithm for optimal assignment
        row_indices, col_indices = linear_sum_assignment(-iou_matrix)

        matches = []
        unmatched_trackers = []
        unmatched_detections = []

        # Check each potential match against thresholds
        for t, d in zip(row_indices, col_indices):
            tb = trackers_boxes[t]
            db = detections_boxes[d]

            # Calculate center distance
            center_tracker = ((tb[0] + tb[2]) / 2.0, (tb[1] + tb[3]) / 2.0)
            center_detection = ((db[0] + db[2]) / 2.0, (db[1] + db[3]) / 2.0)
            dist = math.hypot(
                center_tracker[0] - center_detection[0],
                center_tracker[1] - center_detection[1],
            )

            if iou_matrix[t, d] >= self.iou_threshold or dist < self.distance_threshold:
                matches.append((t, d))
            else:
                unmatched_trackers.append(t)
                unmatched_detections.append(d)

        # Add remaining trackers and detections
        for t in range(trackers_boxes.shape[0]):
            if t not in row_indices:
                unmatched_trackers.append(t)
        for d in range(detections_boxes.shape[0]):
            if d not in col_indices:
                unmatched_detections.append(d)

        return matches, unmatched_trackers, unmatched_detections
