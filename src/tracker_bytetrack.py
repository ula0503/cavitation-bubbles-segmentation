import numpy as np
from scipy.optimize import linear_sum_assignment
import math

import numpy as np
from scipy.optimize import linear_sum_assignment
import math

import numpy as np
from scipy.optimize import linear_sum_assignment
import math


def iou(bbox1, bbox2):
    """
    Вычисляет IoU (Intersection over Union) для двух bounding box.
    Формат bbox: [x1, y1, x2, y2]
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, bbox1[2] - bbox1[0]) * max(0, bbox1[3] - bbox1[1])
    area2 = max(0, bbox2[2] - bbox2[0]) * max(0, bbox2[3] - bbox2[1])
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


class KalmanBoxTracker:
    """
    Отслеживает отдельный объект с использованием полноценного Калмана.
    Состояние: [x, y, s, r, vx, vy, vs]
      x, y – координаты центра,
      s – площадь (масштаб),
      r – соотношение сторон (предполагается относительно стабильным),
      vx, vy, vs – скорости соответствующих параметров.
    """

    count = 0

    def __init__(self, bbox, frame_idx, timestamp, detection=None):
        """
        Инициализация трека по начальному bbox.
        bbox: [x1, y1, x2, y2]
        """
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        # Преобразуем bbox в измерение: [x, y, s, r]
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        x = x1 + w / 2.0
        y = y1 + h / 2.0
        s = w * h
        r = w / (h + 1e-6)

        # Инициализируем состояние: [x, y, s, r, vx, vy, vs]
        self.state = np.array([x, y, s, r, 0, 0, 0], dtype=np.float32)
        # Начальная ковариация. Допустим, относительно точны координаты, но неопределенность в площади и особенно в скорости больше.
        self.P = np.diag([10, 10, 100, 10, 1000, 1000, 1000]).astype(np.float32)

        # Шаг времени по умолчанию (если не задан динамически) — 1 единица (1 кадр)
        dt = 1.0
        # Матрица перехода состояния F для модели постоянной скорости.
        # В дальнейшем будем обновлять компоненты, зависящие от dt.
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

        # Матрица наблюдения H (мы измеряем только [x, y, s, r])
        self.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ],
            dtype=np.float32,
        )

        # Матрица шума процесса Q — увеличим шум для скоростных компонент.
        self.Q = np.diag([0.1, 0.1, 0.1, 0.001, 50, 50, 50]).astype(np.float32)
        # Матрица шума измерения R — предполагаем, что измерения x, y относительно точны, площадь менее точна.
        self.R = np.diag([0.5, 0.5, 10, 0.01]).astype(np.float32)

        self.frame_idx = frame_idx
        self.timestamp = timestamp
        self.time_since_update = 0
        self.history = [bbox]
        self.detection = detection

    def predict(self, dt=None):
        """
        Выполняет предсказание нового состояния.
        Если dt не задан, предполагаем dt = 1.0.
        Если задан, обновляем матрицу перехода F.
        """
        if dt is None:
            dt = 1.0
        # Обновляем dt в матрице перехода F
        self.F[0, 4] = dt
        self.F[1, 5] = dt
        self.F[2, 6] = dt

        self.state = np.dot(self.F, self.state)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        self.time_since_update += 1
        return self.get_state()

    def update(self, bbox, frame_idx, timestamp, detection=None):
        """
        Обновление состояния по новой детекции с использованием стандартного уравнения Калмана.
        dt рассчитывается как разница между текущей временной меткой и предыдущей.
        """
        # Рассчитываем dt по времени между текущей и предыдущей детекцией.
        dt = timestamp - self.timestamp if self.timestamp is not None else 1.0
        if dt <= 0:
            dt = 1.0

        # Обновляем матрицу перехода F с учетом dt
        self.F[0, 4] = dt
        self.F[1, 5] = dt
        self.F[2, 6] = dt

        # Преобразуем bbox в измерение: [x, y, s, r]
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        x = x1 + w / 2.0
        y = y1 + h / 2.0
        s = w * h
        r = w / (h + 1e-6)
        z = np.array([x, y, s, r], dtype=np.float32)

        # Измеренная инновация: y = z - Hx
        y_meas = z - np.dot(self.H, self.state)
        # Инновационная ковариация: S = HPH^T + R
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        # Оптимальное усиление Калмана: K = PH^T S^-1
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        # Обновление состояния: x = x + Ky
        self.state = self.state + np.dot(K, y_meas)
        # Обновление ковариации: P = (I - KH)P
        I = np.eye(self.F.shape[0], dtype=np.float32)
        self.P = np.dot((I - np.dot(K, self.H)), self.P)

        self.time_since_update = 0
        self.frame_idx = frame_idx
        self.timestamp = timestamp
        self.history.append(bbox)
        self.detection = detection

    def get_state(self):
        """
        Возвращает текущий bbox в формате [x1, y1, x2, y2] на основании текущего состояния.
        Производится защита от отрицательных значений для s и r.
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
    def __init__(
        self,
        high_thresh=0.6,
        low_thresh=0.1,
        max_time_lost=10,
        iou_threshold=0.2,
        distance_threshold=50,
    ):
        """
        Инициализация ByteTracker.
          - high_thresh: порог для высокодоверенных детекций.
          - low_thresh: порог для низкодоверенных детекций.
          - max_time_lost: число кадров, в течение которых трек может оставаться без обновления.
          - iou_threshold: порог IoU для сопоставления детекций с треками.
          - distance_threshold: максимальное допустимое расстояние между центрами bbox для сопоставления.
        """
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.max_time_lost = max_time_lost
        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold
        self.trackers = []
        self.finished_tracks = []  # Для хранения завершённых треков

    def update(self, detections, frame_idx, timestamp):
        """
        Обновляет трекер по новым детекциям.
        detections: список словарей с ключами 'bbox', 'confidence', 'mask', и т.д.
        Возвращает словарь активных треков: {tracker_id: tracker_instance}.
        """
        # Разбиваем детекции на высокодоверенные и низкодоверенные
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

        # Предсказываем новое положение для всех треков
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

        # 1. Сопоставляем высокодоверенные детекции с существующими треками
        matches, unmatched_trackers, unmatched_detections = (
            self.associate_detections_to_trackers(predicted_boxes, high_boxes)
        )

        # Обновляем треки, для которых найдено соответствие
        for tracker_idx, detection_idx in matches:
            self.trackers[tracker_idx].update(
                high_boxes[detection_idx],
                frame_idx,
                timestamp,
                detection=high_detections[detection_idx],
            )

        # 2. Для оставшихся треков пытаемся сопоставить низкодоверенные детекции
        if len(unmatched_trackers) > 0 and low_boxes.shape[0] > 0:
            unmatched_predicted = predicted_boxes[unmatched_trackers]
            matches_low, unmatched_trackers_final, unmatched_low = (
                self.associate_detections_to_trackers(unmatched_predicted, low_boxes)
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

        # 3. Для треков, которым не нашли соответствия, увеличиваем счетчик пропусков
        for idx in unmatched_trackers:
            self.trackers[idx].time_since_update += 1

        # 4. Создаем новые треки для высокодоверенных детекций, которым не нашли соответствия
        for detection_idx in unmatched_detections:
            det = high_detections[detection_idx]
            new_tracker = KalmanBoxTracker(
                det["bbox"], frame_idx, timestamp, detection=det
            )
            self.trackers.append(new_tracker)

        # 5. Переносим треки, которые не обновлялись слишком долго, в finished_tracks
        active_trackers = []
        for tracker in self.trackers:
            if tracker.time_since_update > self.max_time_lost:
                self.finished_tracks.append(tracker)
            else:
                active_trackers.append(tracker)
        self.trackers = active_trackers

        # Возвращаем активные треки – считаем активными те, у которых time_since_update <= 1
        active = {
            tracker.id: tracker
            for tracker in self.trackers
            if tracker.time_since_update <= 1
        }
        return active

    def associate_detections_to_trackers(self, trackers_boxes, detections_boxes):
        """
        Сопоставляет детекции с треками с использованием матрицы IoU и комбинированного критерия.
        Возвращает:
            matches: список пар (tracker_idx, detection_idx)
            unmatched_trackers: список индексов треков без сопоставления
            unmatched_detections: список индексов детекций без сопоставления
        """
        if trackers_boxes.shape[0] == 0 or detections_boxes.shape[0] == 0:
            return (
                [],
                list(range(trackers_boxes.shape[0])),
                list(range(detections_boxes.shape[0])),
            )

        # Вычисляем матрицу IoU
        iou_matrix = np.zeros(
            (trackers_boxes.shape[0], detections_boxes.shape[0]), dtype=np.float32
        )
        for t, tb in enumerate(trackers_boxes):
            for d, db in enumerate(detections_boxes):
                iou_matrix[t, d] = iou(tb, db)

        row_indices, col_indices = linear_sum_assignment(-iou_matrix)
        matches = []
        unmatched_trackers = []
        unmatched_detections = []

        # Для каждой пары (tracker, детекция) проверяем, если либо IoU достаточно высокое,
        # либо центры bbox близки (евклидово расстояние меньше distance_threshold), то считаем их совпадающими.
        for t, d in zip(row_indices, col_indices):
            tb = trackers_boxes[t]
            db = detections_boxes[d]
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

        # Добавляем те треки и детекции, которые не попали в алгоритм Хунгера
        for t in range(trackers_boxes.shape[0]):
            if t not in row_indices:
                unmatched_trackers.append(t)
        for d in range(detections_boxes.shape[0]):
            if d not in col_indices:
                unmatched_detections.append(d)
        return matches, unmatched_trackers, unmatched_detections
