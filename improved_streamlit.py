import streamlit as st
import cv2
import tempfile
import os
import uuid
import pandas as pd
import numpy as np
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


def analyze_bubble_statistics(csv_file_path):
    """Анализ статистики пузырьков из CSV файла"""
    df = pd.read_csv(csv_file_path)

    results = []

    for tracker_id, group in df.groupby("tracker_id"):
        if len(group) < 2:
            continue

        track_data = {
            "tracker_id": tracker_id,
            "avg_area": group["area"].mean(),
            "std_area": group["area"].std(),
            "initial_area": group["area"].iloc[0],
            "final_area": group["area"].iloc[-1],
            "max_area": group["area"].max(),
            "min_area": group["area"].min(),
            "avg_speed": group["speed_px_per_sec"].mean(),
            "max_speed": group["speed_px_per_sec"].max(),
            "std_speed": group["speed_px_per_sec"].std(),
            "lifetime": group["timestamp"].max() - group["timestamp"].min(),
            "measurement_count": len(group),
        }

        nonzero_speeds = group[group["speed_px_per_sec"] > 0]["speed_px_per_sec"]
        track_data["avg_nonzero_speed"] = (
            nonzero_speeds.mean() if len(nonzero_speeds) > 0 else 0
        )

        if track_data["initial_area"] > 0:
            track_data["area_change_ratio"] = (
                track_data["final_area"] - track_data["initial_area"]
            ) / track_data["initial_area"]
        else:
            track_data["area_change_ratio"] = 0

        if track_data["avg_speed"] > 0:
            track_data["speed_variation"] = (
                track_data["std_speed"] / track_data["avg_speed"]
            )
        else:
            track_data["speed_variation"] = 0

        results.append(track_data)

    bubble_stats = pd.DataFrame(results).round(3)

    if len(bubble_stats) > 0:
        overall_stats = {
            "total_bubbles": len(bubble_stats),
            "avg_lifetime": bubble_stats["lifetime"].mean(),
            "std_lifetime": bubble_stats["lifetime"].std(),
            "avg_area_all": bubble_stats["avg_area"].mean(),
            "avg_speed_all": bubble_stats["avg_speed"].mean(),
            "avg_nonzero_speed_all": bubble_stats["avg_nonzero_speed"].mean(),
            "max_speed_all": bubble_stats["max_speed"].max(),
        }
    else:
        overall_stats = {}

    return bubble_stats, overall_stats


def save_statistics_report(bubble_stats, overall_stats, output_dir, report_name):
    """Сохраняет финальный отчет в папку"""
    reports_dir = os.path.join(output_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    stats_csv_path = os.path.join(reports_dir, f"{report_name}_detailed_stats.csv")
    bubble_stats.to_csv(stats_csv_path, index=False)

    overall_stats_path = os.path.join(reports_dir, f"{report_name}_overall_stats.csv")
    overall_df = pd.DataFrame([overall_stats])
    overall_df.to_csv(overall_stats_path, index=False)

    report_path = os.path.join(reports_dir, f"{report_name}_summary.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("ОТЧЕТ ПО АНАЛИЗУ КАВИТАЦИОННЫХ ПУЗЫРЬКОВ\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Общее количество пузырьков: {overall_stats['total_bubbles']}\n")
        f.write(f"Среднее время жизни: {overall_stats['avg_lifetime']:.2f} сек\n")
        f.write(
            f"Стандартное отклонение времени жизни: {overall_stats['std_lifetime']:.2f} сек\n"
        )
        f.write(f"Средняя площадь: {overall_stats['avg_area_all']:.1f} px²\n")
        f.write(f"Средняя скорость: {overall_stats['avg_speed_all']:.1f} px/сек\n")
        f.write(
            f"Средняя скорость (без нулей): {overall_stats['avg_nonzero_speed_all']:.1f} px/сек\n"
        )
        f.write(f"Максимальная скорость: {overall_stats['max_speed_all']:.1f} px/сек\n")
        f.write(
            f"\nДата создания отчета: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )

    return stats_csv_path, overall_stats_path, report_path


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
                video_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                video_session_dir = os.path.join(
                    VIDEO_RESULTS_DIR, f"video_{video_timestamp}"
                )
                os.makedirs(video_session_dir, exist_ok=True)

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=os.path.splitext(video_file.name)[1]
                ) as temp_file:
                    temp_file.write(video_file.getvalue())
                    temp_path = temp_file.name

                result_filename = (
                    f"processed_{os.path.splitext(video_file.name)[0]}.mp4"
                )
                result_path = os.path.join(video_session_dir, result_filename)
                data_path = os.path.join(video_session_dir, f"analysis_data.csv")
                charts_dir = os.path.join(video_session_dir, "charts")
                os.makedirs(charts_dir, exist_ok=True)

                try:
                    processing_path, _ = prepare_video_format(
                        temp_path, video_file.name
                    )

                    video_processor = VideoProcessor(model_config.segmentation_model)
                    speed_chart, area_chart = video_processor.process_video(
                        processing_path, result_path, data_path, charts_dir
                    )

                    bubble_stats, overall_stats = analyze_bubble_statistics(data_path)

                    if not bubble_stats.empty:
                        report_name = f"report_{os.path.splitext(video_file.name)[0]}"
                        stats_csv_path, overall_stats_path, report_path = (
                            save_statistics_report(
                                bubble_stats,
                                overall_stats,
                                video_session_dir,
                                report_name,
                            )
                        )

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

                    if processing_path != temp_path:
                        os.unlink(processing_path)

                except Exception as error:
                    st.error(f"Ошибка обработки: {error}")
                finally:
                    os.unlink(temp_path)

    if st.session_state.current_video:
        current_results = st.session_state.processed_files[
            st.session_state.current_video
        ]

        left_column, right_column = st.columns(2)

        with left_column:
            if os.path.exists(current_results["video_output"]):
                st.video(current_results["video_output"])
                st.info(f"Видео сохранено: {current_results['video_output']}")

        with right_column:
            if os.path.exists(current_results["data_file"]):
                data_frame = pd.read_csv(current_results["data_file"])
                st.dataframe(data_frame.head(10))

                bubble_stats, overall_stats = analyze_bubble_statistics(
                    current_results["data_file"]
                )

                if not bubble_stats.empty:
                    st.info("### Статистика пузырьков:")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Всего пузырьков", overall_stats["total_bubbles"])
                        st.metric(
                            "Среднее время жизни",
                            f"{overall_stats['avg_lifetime']:.2f} сек",
                        )
                        st.metric(
                            "Средняя площадь",
                            f"{overall_stats['avg_area_all']:.1f} px²",
                        )
                    with col2:
                        st.metric(
                            "Макс. скорость",
                            f"{overall_stats['max_speed_all']:.1f} px/сек",
                        )
                        st.metric(
                            "Средняя скорость",
                            f"{overall_stats['avg_speed_all']:.1f} px/сек",
                        )
                        st.metric(
                            "Скорость (без нулей)",
                            f"{overall_stats['avg_nonzero_speed_all']:.1f} px/сек",
                        )

                    st.info("### Детальная статистика по пузырькам:")
                    st.dataframe(bubble_stats)

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

    st.write("### Настройка папке для результатов")

    col1, col2 = st.columns(2)
    with col1:
        project_name = st.text_input("Название проекта", "кавитационный_эксперимент")
    with col2:
        concentration = st.text_input("Концентрация раствора", "5%")

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
                combined_data = []

                for index, video in enumerate(multiple_videos):
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=os.path.splitext(video.name)[1]
                    ) as temp_file:
                        temp_file.write(video.getvalue())
                        temp_video_path = temp_file.name

                    video_basename = os.path.splitext(video.name)[0]
                    output_video_path = os.path.join(
                        video_results_dir, f"processed_{video_basename}.mp4"
                    )
                    output_data_path = os.path.join(
                        data_results_dir, f"data_{video_basename}.csv"
                    )

                    try:
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

                        if os.path.exists(output_data_path):
                            video_data = pd.read_csv(output_data_path)
                            video_data["source_video"] = video.name
                            combined_data.append(video_data)
                            os.remove(output_data_path)

                        processing_results.append(
                            {
                                "video_name": video.name,
                                "output_video": output_video_path,
                                "data_file": "включено в общий файл",
                                "status": "Успешно",
                            }
                        )

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

                combined_data_path = os.path.join(
                    data_results_dir, "combined_analysis_data.csv"
                )
                if combined_data:
                    final_combined_data = pd.concat(combined_data, ignore_index=True)
                    final_combined_data.to_csv(combined_data_path, index=False)

                    bubble_stats, overall_stats = analyze_bubble_statistics(
                        combined_data_path
                    )

                    if not bubble_stats.empty:
                        report_name = f"batch_report_{batch_folder_name}"
                        stats_csv_path, overall_stats_path, report_path = (
                            save_statistics_report(
                                bubble_stats, overall_stats, main_dir, report_name
                            )
                        )

                successful_processing = len(
                    [r for r in processing_results if r["status"] == "Успешно"]
                )
                st.success(
                    f"Успешно обработано: {successful_processing} из {len(processing_results)}"
                )
                st.success(f"Все результаты сохранены в папке: {main_dir}")

                results_table = pd.DataFrame(processing_results)
                st.dataframe(results_table)

                if combined_data and os.path.exists(combined_data_path):
                    st.success("### Анализ статистики пузырьков")

                    if not bubble_stats.empty:
                        st.info("### Детальная статистика по пузырькам:")
                        st.dataframe(bubble_stats)

                        st.info("### Общая статистика по всем пузырькам:")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Всего пузырьков", overall_stats["total_bubbles"])
                            st.metric(
                                "Среднее время жизни",
                                f"{overall_stats['avg_lifetime']:.2f} сек",
                            )
                            st.metric(
                                "Стд. время жизни",
                                f"{overall_stats['std_lifetime']:.2f} сек",
                            )
                        with col2:
                            st.metric(
                                "Средняя площадь",
                                f"{overall_stats['avg_area_all']:.1f} px²",
                            )
                            st.metric(
                                "Макс. скорость",
                                f"{overall_stats['max_speed_all']:.1f} px/сек",
                            )
                        with col3:
                            st.metric(
                                "Средняя скорость",
                                f"{overall_stats['avg_speed_all']:.1f} px/сек",
                            )
                            st.metric(
                                "Скорость (без нулей)",
                                f"{overall_stats['avg_nonzero_speed_all']:.1f} px/сек",
                            )
