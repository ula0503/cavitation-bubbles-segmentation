import cv2
import numpy as np
import sys
import os

# Добавляем путь к src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.segmentation import YoloSegmenter
from src.tracker_bytetrack import ByteTracker, KalmanBoxTracker


def debug_tracker_history():
    """Отладочный тест для проверки структуры истории трекера"""

    print("=" * 60)
    print("ОТЛАДКА СТРУКТУРЫ ТРЕКЕРА")
    print("=" * 60)

    # 1. Создаем тестовый трекер
    print("\n1. Создаем тестовый KalmanBoxTracker")
    test_bbox = [100, 100, 150, 150]
    tracker = KalmanBoxTracker(test_bbox, frame_idx=0, timestamp=0.0)

    print(f"   ID трекера: {tracker.id}")
    print(f"   Длина истории после инициализации: {len(tracker.history)}")
    print(f"   Тип истории: {type(tracker.history)}")

    if tracker.history:
        print(f"   Первый элемент истории: {tracker.history[0]}")
        print(f"   Тип первого элемента: {type(tracker.history[0])}")

        # Проверяем структуру
        if isinstance(tracker.history[0], dict):
            print(f"   ✓ История содержит словари")
            print(f"   Ключи в словаре: {list(tracker.history[0].keys())}")
            if "bbox" in tracker.history[0]:
                print(f"   ✓ Ключ 'bbox' присутствует")
                print(f"   Значение 'bbox': {tracker.history[0]['bbox']}")
        else:
            print(f"   ✗ История НЕ содержит словари!")
            print(f"   Это список: {type(tracker.history[0])}")

    # 2. Делаем predict
    print("\n2. Вызываем predict()")
    predicted_bbox = tracker.predict(dt=1.0)
    print(f"   Длина истории после predict: {len(tracker.history)}")
    print(f"   Последний элемент истории: {tracker.history[-1]}")

    # 3. Делаем update
    print("\n3. Вызываем update()")
    new_bbox = [110, 110, 160, 160]
    tracker.update(new_bbox, frame_idx=1, timestamp=0.2)
    print(f"   Длина истории после update: {len(tracker.history)}")

    # 4. Проверяем все элементы истории
    print("\n4. Проверяем все элементы истории:")
    for i, item in enumerate(tracker.history):
        print(f"   [{i}] Тип: {type(item)}")
        if isinstance(item, dict):
            print(f"       Ключи: {list(item.keys())}")
            if "bbox" in item:
                print(f"       bbox: {item['bbox']}")
        else:
            print(f"       Значение: {item}")

    # 5. Тестируем метод из video_processing
    print("\n5. Тестируем _calculate_speed_from_history (упрощенный):")

    def test_speed_calculation(tracker_obj, video_fps=5.0):
        if len(tracker_obj.history) < 2:
            print("   История слишком короткая")
            return

        history_prev = tracker_obj.history[-2]
        history_curr = tracker_obj.history[-1]

        print(f"   history_prev тип: {type(history_prev)}")
        print(f"   history_curr тип: {type(history_curr)}")

        try:
            # Пробуем извлечь bbox
            if isinstance(history_prev, dict) and "bbox" in history_prev:
                bbox_prev = history_prev["bbox"]
                print(f"   ✓ Успешно извлекли bbox_prev: {bbox_prev}")
            else:
                print(f"   ✗ history_prev не словарь или нет ключа 'bbox'")

            if isinstance(history_curr, dict) and "bbox" in history_curr:
                bbox_curr = history_curr["bbox"]
                print(f"   ✓ Успешно извлекли bbox_curr: {bbox_curr}")
            else:
                print(f"   ✗ history_curr не словарь или нет ключа 'bbox'")

        except Exception as e:
            print(f"   ✗ Ошибка при извлечении bbox: {e}")

    test_speed_calculation(tracker)

    print("\n" + "=" * 60)
    print("ОТЛАДКА ЗАВЕРШЕНА")
    print("=" * 60)


def debug_actual_video_processing():
    """Отладка на реальном кадре"""
    print("\n" + "=" * 60)
    print("ОТЛАДКА НА РЕАЛЬНОМ КАДРЕ")
    print("=" * 60)

    # Создаем тестовый кадр
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Инициализируем сегментатор (нужна модель)
    try:
        from model_config import model_config

        if model_config.check_models():
            segmenter = YoloSegmenter(model_config.segmentation_model)

            # Запускаем сегментацию
            print("\nЗапускаем сегментацию тестового кадра...")
            detections = segmenter.segment_frame(test_frame)
            print(f"Найдено детекций: {len(detections)}")

            # Инициализируем трекер
            tracker = ByteTracker()

            # Обновляем трекер
            print("Запускаем трекинг...")
            tracked_objects = tracker.update(detections, frame_idx=0, timestamp=0.0)
            print(f"Треков создано: {len(tracked_objects)}")

            if tracked_objects:
                # Берем первый трекер
                first_tracker_id = list(tracked_objects.keys())[0]
                first_tracker = tracked_objects[first_tracker_id]

                print(f"\nПроверяем первый трекер (ID: {first_tracker.id}):")
                print(f"  Тип трекера: {type(first_tracker)}")
                print(f"  Длина истории: {len(first_tracker.history)}")

                if first_tracker.history:
                    print(f"  Первый элемент истории: {first_tracker.history[0]}")
                    print(f"  Тип элемента: {type(first_tracker.history[0])}")
        else:
            print("Модель не найдена, пропускаем тест сегментации")

    except ImportError as e:
        print(f"Не удалось импортировать model_config: {e}")
    except Exception as e:
        print(f"Ошибка при тесте: {e}")


if __name__ == "__main__":
    print("Запуск отладки структуры трекера...")
    debug_tracker_history()

    # Раскомментируйте для теста с реальной сегментацией
    # debug_actual_video_processing()
