"""Video processing module for bubble tracking."""

import csv
import math
import os
from typing import Optional, Tuple, List, Dict, Any
import cv2
import numpy as np
from .segmentation import YoloSegmenter
from .tracker_bytetrack import ByteTracker
from .utils import draw_mask, compute_centroid, euclidean_distance


class VideoProcessor:
    """Process video frames for bubble detection and tracking."""

    def __init__(self, model_path: str) -> None:
        """Initialize video processor with segmentation model.

        Args:
            model_path: Path to YOLO segmentation model weights.
        """
        self.segmenter = YoloSegmenter(model_path)
        self.tracker = ByteTracker(
            high_thresh=0.5,  # Confidence threshold for high-quality detections
            low_thresh=0.1,  # Confidence threshold for low-quality detections
            max_time_lost=30,  # Max frames to keep track without updates
            iou_threshold=0.2,  # IoU threshold for detection association
            distance_threshold=50,  # Euclidean distance threshold for detection association
        )

    def _calculate_speed_from_history(self, tracker: Any, video_fps: float) -> float:
        """Calculate speed in pixels per second from tracker history.

        Args:
            tracker: KalmanBoxTracker instance with bbox history.
            video_fps: Frames per second of the video.

        Returns:
            Speed in pixels per second. Returns 0.0 if insufficient history.
        """
        if len(tracker.history) < 2:
            return 0.0

        # Get last two bboxes from history
        bbox_prev = tracker.history[-2]  # [x1, y1, x2, y2]
        bbox_curr = tracker.history[-1]  # [x1, y1, x2, y2]

        # Calculate centroids
        centroid_prev = compute_centroid(bbox_prev)
        centroid_curr = compute_centroid(bbox_curr)

        # Calculate pixel distance
        distance_px = euclidean_distance(centroid_prev, centroid_curr)

        # Time between frames
        dt_real_seconds = 1.0 / video_fps

        # Speed = distance / time
        speed_px_per_sec = distance_px / dt_real_seconds

        return speed_px_per_sec

    def process_video(
        self,
        input_video_path: str,
        output_video_path: str,
        csv_path: str,
    ) -> Tuple[None, None]:
        """Process video and track bubbles.

        Args:
            input_video_path: Path to input video file.
            output_video_path: Path to save annotated video.
            csv_path: Path to save tracking data CSV.

        Returns:
            Tuple of (None, None) for backward compatibility.
        """
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print("Error: Could not open video file.")
            return None, None

        # Get FPS from video
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 5.0
            print(f"FPS not detected, using default: {fps}")
        else:
            print(f"Original video FPS: {fps}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        csv_file = open(csv_path, mode="w", newline="")
        csv_writer = csv.writer(csv_file)

        # Write CSV headers with all tracking data
        csv_writer.writerow(
            [
                "tracker_id",
                "frame_idx",
                "timestamp",
                "centroid_x",
                "centroid_y",
                "area",
                "class",
                "speed_px_per_sec",
                "confidence",
                "track_length",
                "frames_lost",
                "bbox_x1",
                "bbox_y1",
                "bbox_x2",
                "bbox_y2",
                "kalman_area",
                "velocity_x",
                "velocity_y",
                "displacement",
                "trajectory_angle",
            ]
        )

        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx / fps
            detections = self.segmenter.segment_frame(frame)
            tracked_objects = self.tracker.update(detections, frame_idx, timestamp)
            annotated_frame = frame.copy()

            for tracker in tracked_objects.values():
                bbox = tracker.get_state()
                cx = int((bbox[0] + bbox[2]) / 2)
                cy = int((bbox[1] + bbox[3]) / 2)

                detection = tracker.detection
                if detection is None:
                    continue

                mask = detection.get("mask")
                detection_class = detection.get("class")
                color = (0, 255, 0) if detection_class == 0 else (0, 0, 255)

                # Draw bubble ID only
                cv2.putText(
                    annotated_frame,
                    f"ID: {tracker.id}",
                    (cx, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

                if mask is not None:
                    annotated_frame = draw_mask(annotated_frame, mask, color)

                # Calculate metrics
                speed_px_per_sec = self._calculate_speed_from_history(tracker, fps)
                area = tracker.state[2]

                # Calculate displacement and trajectory angle for CSV only
                displacement = 0.0
                angle = 0.0
                if len(tracker.history) >= 2:
                    prev_bbox = tracker.history[-2]
                    prev_cx = (prev_bbox[0] + prev_bbox[2]) / 2
                    prev_cy = (prev_bbox[1] + prev_bbox[3]) / 2
                    displacement = math.hypot(cx - prev_cx, cy - prev_cy)

                    # Calculate angle in degrees (0° = right, 90° = up)
                    if displacement > 0:
                        # Inverted Y for image coordinates
                        angle = math.degrees(math.atan2(prev_cy - cy, cx - prev_cx))
                        angle = (angle + 360) % 360  # Normalize to 0-360

                csv_writer.writerow(
                    [
                        tracker.id,
                        frame_idx,
                        timestamp,
                        cx,
                        cy,
                        area,
                        detection_class,
                        speed_px_per_sec,
                        detection.get("confidence", 0.0),
                        len(tracker.history),
                        tracker.time_since_update,
                        bbox[0],  # x1
                        bbox[1],  # y1
                        bbox[2],  # x2
                        bbox[3],  # y2
                        tracker.state[2],  # kalman area
                        tracker.state[4],  # velocity x
                        tracker.state[5],  # velocity y
                        displacement,
                        angle,
                    ]
                )

            out.write(annotated_frame)
            frame_idx += 1

        cap.release()
        out.release()
        csv_file.close()

        return None, None
