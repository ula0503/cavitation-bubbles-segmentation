import cv2
import csv
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from .segmentation import YoloSegmenter
from .tracker_bytetrack import ByteTracker
from .utils import draw_mask, compute_centroid, euclidean_distance


class VideoProcessor:
    def __init__(self, model_path: str):
        self.segmenter = YoloSegmenter(model_path)
        self.tracker = ByteTracker(
            high_thresh=0.5,
            low_thresh=0.1,
            max_time_lost=10,
            iou_threshold=0.2,
            distance_threshold=50,
        )

    def _calculate_speed_from_history(self, tracker, video_fps):
        """
        Вычисляет скорость в реальных пикселях/секунду.
        История содержит bbox напрямую (старая версия).
        """
        if len(tracker.history) < 2:
            return 0.0

        # Берем последние два bbox из истории (они уже bbox, не словари!)
        bbox_prev = tracker.history[-2]  # [x1, y1, x2, y2]
        bbox_curr = tracker.history[-1]  # [x1, y1, x2, y2]

        # Вычисляем центроиды
        centroid_prev = compute_centroid(bbox_prev)
        centroid_curr = compute_centroid(bbox_curr)

        # Расстояние в пикселях между центрами
        distance_px = euclidean_distance(centroid_prev, centroid_curr)

        # Время между кадрами
        dt_real_seconds = 1.0 / video_fps

        # Скорость = расстояние / время
        speed_px_per_sec = distance_px / dt_real_seconds

        return speed_px_per_sec

    def process_video(
        self,
        input_video_path: str,
        output_video_path: str,
        csv_path: str,
        hist_folder: str,
    ):
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print("Ошибка: не удалось открыть видеофайл.")
            return None, None

        # Берем REAL FPS из исходного видео
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            # Если FPS не определен, используем 5.0 (ваш реальный FPS)
            fps = 5.0
            print(f"FPS не определен, используем: {fps}")
        else:
            print(f"Исходный FPS видео: {fps}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # Записываем с ТЕМ ЖЕ FPS, что и исходное видео
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        csv_file = open(csv_path, mode="w", newline="")
        csv_writer = csv.writer(csv_file)
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

                mask = detection.get("mask", None)
                detection_class = detection.get("class", None)
                color = (0, 255, 0) if detection_class == 0 else (0, 0, 255)

                # ТОЛЬКО ID на кадре
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

                # Вычисляем скорость для записи в CSV
                speed_px_per_sec = self._calculate_speed_from_history(tracker, fps)

                # Площадь из состояния Калмана (уже в пикселях^2)
                area = tracker.state[2]

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
                    ]
                )

            out.write(annotated_frame)
            frame_idx += 1

        cap.release()
        out.release()
        csv_file.close()

        speed_hist_file, area_hist_file = self.generate_histograms(hist_folder, fps)
        return speed_hist_file, area_hist_file

    def generate_histograms(self, hist_folder: str, fps: float):
        all_tracks = self.tracker.finished_tracks + self.tracker.trackers
        if not all_tracks:
            return None, None

        sorted_tracks = sorted(all_tracks, key=lambda tr: len(tr.history), reverse=True)
        top20 = sorted_tracks[:20]
        speeds = []
        areas = []

        for tr in top20:
            # Вычисляем скорость тем же методом
            speed_px_per_sec = self._calculate_speed_from_history(tr, fps)
            speeds.append(speed_px_per_sec)
            areas.append(tr.state[2])

        # Гистограмма скорости
        plt.figure(figsize=(10, 6))
        plt.hist(speeds, bins=15, color="blue", alpha=0.7, edgecolor="black")
        plt.title(f"Гистограмма скорости пузырьков (FPS: {fps:.1f})")
        plt.xlabel("Скорость (пикселей/секунду)")
        plt.ylabel("Количество пузырьков")
        plt.grid(True, alpha=0.3)
        speed_hist_file = os.path.join(hist_folder, "histogram_speed.png")
        plt.savefig(speed_hist_file, dpi=150, bbox_inches="tight")
        plt.close()

        # Гистограмма площади
        plt.figure(figsize=(10, 6))
        plt.hist(areas, bins=15, color="green", alpha=0.7, edgecolor="black")
        plt.title("Гистограмма площади пузырьков")
        plt.xlabel("Площадь (пикселей²)")
        plt.ylabel("Количество пузырьков")
        plt.grid(True, alpha=0.3)
        area_hist_file = os.path.join(hist_folder, "histogram_area.png")
        plt.savefig(area_hist_file, dpi=150, bbox_inches="tight")
        plt.close()

        return speed_hist_file, area_hist_file
