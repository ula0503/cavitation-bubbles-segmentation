import numpy as np
from src.utils import compute_centroid, euclidean_distance

class Bubble:
    def __init__(self, bubble_id: int, detection: dict, frame_idx: int, timestamp: float):
        """
        Инициализация нового пузырька с уникальным ID и сохранением первой детекции.
        """
        self.id = bubble_id
        self.history = []  # История обновлений: список словарей с данными за каждый кадр
        self.missed_frames = 0  # Счётчик пропущенных кадров (если объект не найден)
        self.update(detection, frame_idx, timestamp)

    def update(self, detection: dict, frame_idx: int, timestamp: float):
        """
        Обновление информации о пузырьке новой детекцией. Рассчитывается скорость (пикселей/сек).
        """
        bbox = detection.get('bbox', None)
        mask = detection.get('mask', None)
        detection_class = detection.get('class', None)
        centroid = compute_centroid(bbox) if bbox is not None else None

        # Вычисляем площадь: по маске (если есть) или по bbox
        if mask is not None:
            area = float(np.sum(mask > 0))
        elif bbox is not None:
            x1, y1, x2, y2 = bbox
            area = abs((x2 - x1) * (y2 - y1))
        else:
            area = 0

        # Вычисляем скорость (если имеется предыдущая информация)
        speed = 0.0
        if self.history:
            prev = self.history[-1]
            prev_centroid = prev.get('centroid')
            dt = timestamp - prev.get('timestamp', timestamp)
            if centroid is not None and prev_centroid is not None and dt > 0:
                speed = euclidean_distance(centroid, prev_centroid) / dt

        self.history.append({
            'frame_idx': frame_idx,
            'timestamp': timestamp,
            'centroid': centroid,
            'bbox': bbox,
            'area': area,
            'class': detection_class,
            'mask': mask,
            'speed': speed
        })
        self.missed_frames = 0

    def last_position(self):
        """
        Возвращает последний известный центр и bbox.
        """
        if self.history:
            return self.history[-1]['centroid'], self.history[-1]['bbox']
        return None, None

class BubbleTracker:
    def __init__(self, distance_threshold: float = 50.0, max_missed: int = 5):
        """
        Инициализация трекера:
          - distance_threshold: максимальное расстояние (в пикселях) для сопоставления детекции с существующим пузырьком;
          - max_missed: число кадров, в течение которых объект может не детектироваться, прежде чем его удалить.
        """
        self.distance_threshold = distance_threshold
        self.max_missed = max_missed
        self.tracked_bubbles = {}  # Словарь: bubble_id -> Bubble
        self.next_id = 0

    def update(self, detections: list, frame_idx: int, timestamp: float):
        """
        Обновляет состояние трекера с детекциями текущего кадра.
        Возвращает словарь текущих активных пузырьков.
        """
        # Вычисляем центры для каждой детекции
        detection_centroids = []
        for det in detections:
            bbox = det.get('bbox', None)
            centroid = compute_centroid(bbox) if bbox is not None else None
            detection_centroids.append(centroid)

        # Собираем последние центры уже отслеживаемых пузырьков
        tracked_ids = list(self.tracked_bubbles.keys())
        tracked_centroids = []
        for tid in tracked_ids:
            cent, _ = self.tracked_bubbles[tid].last_position()
            tracked_centroids.append(cent)

        used_detection_indices = set()

        # Жадное сопоставление: для каждого отслеживаемого объекта ищем ближайшую детекцию
        for tid, track_centroid in zip(tracked_ids, tracked_centroids):
            if track_centroid is None:
                continue
            min_dist = float('inf')
            min_idx = -1
            for i, det_centroid in enumerate(detection_centroids):
                if i in used_detection_indices or det_centroid is None:
                    continue
                dist = euclidean_distance(track_centroid, det_centroid)
                if dist < min_dist:
                    min_dist = dist
                    min_idx = i
            if min_idx != -1 and min_dist < self.distance_threshold:
                # Обновляем существующий пузырёк
                self.tracked_bubbles[tid].update(detections[min_idx], frame_idx, timestamp)
                used_detection_indices.add(min_idx)
            else:
                # Если сопоставление не найдено – увеличиваем счётчик пропущенных кадров
                self.tracked_bubbles[tid].missed_frames += 1

        # Создаём новые объекты для оставшихся детекций
        for i, det in enumerate(detections):
            if i not in used_detection_indices:
                bubble = Bubble(self.next_id, det, frame_idx, timestamp)
                self.tracked_bubbles[self.next_id] = bubble
                self.next_id += 1

        # Удаляем объекты, которые пропустили слишком много кадров
        remove_ids = [tid for tid, bubble in self.tracked_bubbles.items() if bubble.missed_frames > self.max_missed]
        for tid in remove_ids:
            del self.tracked_bubbles[tid]

        return self.tracked_bubbles