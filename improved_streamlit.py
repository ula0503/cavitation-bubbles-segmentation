import streamlit as st
import cv2
import tempfile
import os
import uuid
import pandas as pd
from datetime import datetime

# Настройка страницы
st.set_page_config(layout="wide")
st.title("Анализ кавитационных пузырьков")

# Инициализация состояния
if "processed_files" not in st.session_state:
    st.session_state.processed_files = {}
if "current_video" not in st.session_state:
    st.session_state.current_video = None

# Импорт компонентов
from src.video_processing import VideoProcessor
from model_config import model_config

# Проверка доступности модели
if not model_config.check_models():
    st.error("Модель не найдена")
    st.stop()

# Получение информации о модели
model_filename = os.path.basename(model_config.segmentation_model)
model_name = os.path.splitext(model_filename)[0].replace("_", " ").title()

st.success("Модель загружена")
st.info(f"Модель: {model_name}")
st.info(f"Файл: {model_filename}")


def convert_video_to_mp4(input_path, output_path):
    """Конвертация видео в MP4"""
    video_capture = cv2.VideoCapture(input_path)

    if not video_capture.isOpened():
        raise RuntimeError(f"Ошибка открытия видео: {input_path}")

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    while True:
        success, frame = video_capture.read()
        if not success:
            break
        video_writer.write(frame)

    video_capture.release()
    video_writer.release()
    return output_path


def prepare_video_format(input_path, filename):
    """Подготовка видео к обработке"""
    file_extension = os.path.splitext(filename)[1].lower()

    if file_extension == ".mp4":
        return input_path, filename

    mp4_filename = os.path.splitext(filename)[0] + ".mp4"
    temp_mp4_path = f"temp_{uuid.uuid4().hex[:8]}.mp4"

    st.info(f"Конвертация {file_extension.upper()} в MP4")
    convert_video_to_mp4(input_path, temp_mp4_path)

    return temp_mp4_path, mp4_filename


# Основные папки для результатов
VIDEO_RESULTS_DIR = "video_results"
BATCH_RESULTS_DIR = "batch_processing"
os.makedirs(VIDEO_RESULTS_DIR, exist_ok=True)
os.makedirs(BATCH_RESULTS_DIR, exist_ok=True)

# Организация интерфейса вкладками
single_tab, batch_tab = st.tabs(["Обработка одного видео", "Пакетная обработка"])

with single_tab:
    video_file = st.file_uploader("Выберите видеофайл", type=["mp4", "avi", "mov"])

    if video_file:
        st.video(video_file.getvalue())

        if st.button("Запустить обработку"):
            with st.spinner("Обработка видео..."):
                # Создаем папку для этого видео
                video_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                video_session_dir = os.path.join(
                    VIDEO_RESULTS_DIR, f"video_{video_timestamp}"
                )
                os.makedirs(video_session_dir, exist_ok=True)

                # Сохранение временного файла
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=os.path.splitext(video_file.name)[1]
                ) as temp_file:
                    temp_file.write(video_file.getvalue())
                    temp_path = temp_file.name

                # Настройка путей для результатов
                result_filename = (
                    f"processed_{os.path.splitext(video_file.name)[0]}.mp4"
                )
                result_path = os.path.join(video_session_dir, result_filename)
                data_path = os.path.join(video_session_dir, f"analysis_data.csv")
                charts_dir = os.path.join(video_session_dir, "charts")
                os.makedirs(charts_dir, exist_ok=True)

                try:
                    # Подготовка и обработка видео
                    processing_path, _ = prepare_video_format(
                        temp_path, video_file.name
                    )

                    video_processor = VideoProcessor(model_config.segmentation_model)
                    speed_chart, area_chart = video_processor.process_video(
                        processing_path, result_path, data_path, charts_dir
                    )

                    # Сохранение результатов
                    st.session_state.processed_files[video_file.name] = {
                        "video_output": result_path,
                        "data_file": data_path,
                        "speed_chart": speed_chart,
                        "area_chart": area_chart,
                        "output_name": result_filename,
                        "session_dir": video_session_dir,
                    }
                    st.session_state.current_video = video_file.name

                    st.success(
                        f"Обработка завершена! Результаты в папке: {video_session_dir}"
                    )

                    # Очистка временных файлов
                    if processing_path != temp_path:
                        os.unlink(processing_path)

                except Exception as error:
                    st.error(f"Ошибка обработки: {error}")
                finally:
                    os.unlink(temp_path)

    # Отображение результатов
    if st.session_state.current_video:
        current_results = st.session_state.processed_files[
            st.session_state.current_video
        ]

        left_column, right_column = st.columns(2)

        with left_column:
            if os.path.exists(current_results["video_output"]):
                st.video(current_results["video_output"])
                with open(current_results["video_output"], "rb") as video_file:
                    st.download_button(
                        "Скачать обработанное видео",
                        video_file,
                        current_results["output_name"],
                    )

        with right_column:
            if os.path.exists(current_results["data_file"]):
                data_frame = pd.read_csv(current_results["data_file"])
                st.dataframe(data_frame.head(10))
                with open(current_results["data_file"], "rb") as data_file:
                    st.download_button(
                        "Скачать данные анализа",
                        data_file,
                        f"analysis_{st.session_state.current_video}.csv",
                    )

            if current_results["speed_chart"] and os.path.exists(
                current_results["speed_chart"]
            ):
                st.image(current_results["speed_chart"], caption="Скорость пузырьков")

            if current_results["area_chart"] and os.path.exists(
                current_results["area_chart"]
            ):
                st.image(current_results["area_chart"], caption="Площадь пузырьков")

with batch_tab:
    st.subheader("Обработка нескольких видео")

    # Настройка папки для пакетной обработки
    st.write("### Настройка папки для результатов")

    col1, col2 = st.columns(2)
    with col1:
        project_name = st.text_input("Название проекта", "кавитационный_эксперимент")
    with col2:
        concentration = st.text_input("Концентрация раствора", "5%")

    # Автоматическое название папки
    batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    suggested_folder = f"{project_name}_{concentration}_{batch_timestamp}"
    batch_folder_name = st.text_input(
        "Название папки для результатов", suggested_folder
    )

    multiple_videos = st.file_uploader(
        "Выберите видеофайлы для обработки",
        type=["mp4", "avi", "mov"],
        accept_multiple_files=True,
    )

    if st.button("Запустить пакетную обработку"):
        if multiple_videos and batch_folder_name:
            with st.spinner(f"Обработка {len(multiple_videos)} видео..."):
                # Создание структуры папок
                main_dir = os.path.join(BATCH_RESULTS_DIR, batch_folder_name)
                video_results_dir = os.path.join(main_dir, "processed_videos")
                data_results_dir = os.path.join(main_dir, "analysis_data")
                chart_results_dir = os.path.join(main_dir, "statistics_charts")

                for directory in [
                    video_results_dir,
                    data_results_dir,
                    chart_results_dir,
                ]:
                    os.makedirs(directory, exist_ok=True)

                progress_indicator = st.progress(0)
                processing_results = []

                for index, video in enumerate(multiple_videos):
                    # Временное сохранение файла
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=os.path.splitext(video.name)[1]
                    ) as temp_file:
                        temp_file.write(video.getvalue())
                        temp_video_path = temp_file.name

                    # Настройка путей вывода
                    video_basename = os.path.splitext(video.name)[0]
                    output_video_path = os.path.join(
                        video_results_dir, f"processed_{video_basename}.mp4"
                    )
                    output_data_path = os.path.join(
                        data_results_dir, f"data_{video_basename}.csv"
                    )

                    try:
                        # Подготовка и обработка
                        processing_path, _ = prepare_video_format(
                            temp_video_path, video.name
                        )

                        processor = VideoProcessor(model_config.segmentation_model)
                        speed_chart, area_chart = processor.process_video(
                            processing_path,
                            output_video_path,
                            output_data_path,
                            chart_results_dir,
                        )

                        processing_results.append(
                            {
                                "video_name": video.name,
                                "output_video": output_video_path,
                                "data_file": output_data_path,
                                "status": "Успешно",
                            }
                        )

                        # Очистка временных файлов
                        if processing_path != temp_video_path:
                            os.unlink(processing_path)

                    except Exception as error:
                        processing_results.append(
                            {
                                "video_name": video.name,
                                "status": f"Ошибка: {str(error)}",
                            }
                        )
                    finally:
                        os.unlink(temp_video_path)

                    progress_indicator.progress((index + 1) / len(multiple_videos))

                # Отчет о результатах
                successful_processing = len(
                    [r for r in processing_results if r["status"] == "Успешно"]
                )
                st.success(
                    f"Успешно обработано: {successful_processing} из {len(processing_results)}"
                )
                st.success(f"Все результаты сохранены в папке: {main_dir}")

                results_table = pd.DataFrame(processing_results)
                st.dataframe(results_table)

                # Показ структуры папок
                st.info("### Структура созданных папок:")
                st.code(
                    f"""
{batch_folder_name}/
├── processed_videos/      # Обработанные видеофайлы
├── analysis_data/         # CSV файлы с данными
└── statistics_charts/     # Графики и гистограммы
                """
                )
