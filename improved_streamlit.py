"""Streamlit application for cavitation bubble analysis."""

import os
import tempfile
import uuid
from datetime import datetime
from typing import Tuple, Dict, Any, List

import streamlit as st
import cv2
import pandas as pd
import numpy as np

from src.video_processing import VideoProcessor
from model_config import model_config


st.set_page_config(layout="wide")
st.title("Cavitation Bubble Analysis")

if "processed_files" not in st.session_state:
    st.session_state.processed_files = {}
if "current_video" not in st.session_state:
    st.session_state.current_video = None

if not model_config.check_models():
    st.error("Model not found")
    st.stop()

model_filename = os.path.basename(model_config.segmentation_model)
model_name = os.path.splitext(model_filename)[0].replace("_", " ").title()

st.success("Model loaded successfully")
st.info(f"Model: {model_name}")
st.info(f"File: {model_filename}")


def convert_video_to_mp4(input_path: str, output_path: str) -> str:
    """Convert video to MP4 format."""
    video_capture = cv2.VideoCapture(input_path)

    if not video_capture.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

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


def prepare_video_format(input_path: str, filename: str) -> Tuple[str, str]:
    """Prepare video for processing (convert to MP4 if needed)."""
    file_extension = os.path.splitext(filename)[1].lower()

    if file_extension == ".mp4":
        return input_path, filename

    mp4_filename = os.path.splitext(filename)[0] + ".mp4"
    temp_mp4_path = f"temp_{uuid.uuid4().hex[:8]}.mp4"

    st.info(f"Converting {file_extension.upper()} to MP4")
    convert_video_to_mp4(input_path, temp_mp4_path)

    return temp_mp4_path, mp4_filename


def analyze_bubble_statistics(
    csv_file_path: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Analyze bubble statistics from CSV file."""
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

        if "displacement" in group.columns and "centroid_x" in group.columns:
            # Calculate total displacement (distance from first to last point)
            first = group.iloc[0]
            last = group.iloc[-1]
            total_disp = np.sqrt(
                (last["centroid_x"] - first["centroid_x"]) ** 2
                + (last["centroid_y"] - first["centroid_y"]) ** 2
            )

            # Path length is sum of all displacements between frames
            path_len = group["displacement"].sum()

            # Straightness = total displacement / path length
            straightness = total_disp / path_len if path_len > 0 else 1.0

            track_data.update(
                {
                    "total_displacement": round(total_disp, 3),
                    "avg_displacement": group["displacement"].mean(),
                    "max_displacement": group["displacement"].max(),
                    "avg_angle": group["trajectory_angle"].mean(),
                    "path_length": round(path_len, 3),
                    "straightness": round(straightness, 3),
                    "avg_confidence": group["confidence"].mean(),
                    "total_frames_lost": group["frames_lost"].sum(),
                }
            )

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

        if "avg_displacement" in bubble_stats.columns:
            overall_stats.update(
                {
                    "avg_path_length": bubble_stats["path_length"].mean(),
                    "avg_straightness": bubble_stats["straightness"].mean(),
                    "avg_confidence": bubble_stats["avg_confidence"].mean(),
                    "total_tracked_frames": df["frame_idx"].nunique(),
                }
            )
    else:
        overall_stats = {}

    return bubble_stats, overall_stats


def save_statistics_report(
    bubble_stats: pd.DataFrame,
    overall_stats: Dict[str, Any],
    output_dir: str,
    report_name: str,
) -> Tuple[str, str, str]:
    """Save final statistics report to folder."""
    reports_dir = os.path.join(output_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    stats_csv_path = os.path.join(reports_dir, f"{report_name}_detailed_stats.csv")
    bubble_stats.to_csv(stats_csv_path, index=False)

    overall_stats_path = os.path.join(reports_dir, f"{report_name}_overall_stats.csv")
    overall_df = pd.DataFrame([overall_stats])
    overall_df.to_csv(overall_stats_path, index=False)

    report_path = os.path.join(reports_dir, f"{report_name}_summary.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("CAVITATION BUBBLES TRACKING REPORT\n")
        f.write("====================================\n\n")
        f.write(f"Total bubbles tracked: {overall_stats['total_bubbles']}\n")
        f.write(f"Average lifetime: {overall_stats['avg_lifetime']:.2f} sec\n")
        f.write(f"Lifetime std deviation: {overall_stats['std_lifetime']:.2f} sec\n")
        f.write(f"Average area: {overall_stats['avg_area_all']:.1f} px²\n")
        f.write(f"Average speed: {overall_stats['avg_speed_all']:.1f} px/sec\n")
        f.write(
            f"Average speed (non-zero): {overall_stats['avg_nonzero_speed_all']:.1f} px/sec\n"
        )
        f.write(f"Maximum speed: {overall_stats['max_speed_all']:.1f} px/sec\n")

        if "avg_path_length" in overall_stats:
            f.write(f"Average path length: {overall_stats['avg_path_length']:.1f} px\n")
            f.write(f"Average straightness: {overall_stats['avg_straightness']:.3f}\n")
            f.write(f"Average confidence: {overall_stats['avg_confidence']:.3f}\n")
            f.write(f"Total tracked frames: {overall_stats['total_tracked_frames']}\n")

        f.write(f"\nReport generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    return stats_csv_path, overall_stats_path, report_path


# Main results directories
VIDEO_RESULTS_DIR = "video_results"
BATCH_RESULTS_DIR = "batch_processing"
os.makedirs(VIDEO_RESULTS_DIR, exist_ok=True)
os.makedirs(BATCH_RESULTS_DIR, exist_ok=True)

single_tab, batch_tab = st.tabs(["Single Video Processing", "Batch Processing"])

with single_tab:
    """Single video processing tab."""
    video_file = st.file_uploader("Select video file", type=["mp4", "avi", "mov"])

    if video_file:
        st.video(video_file.getvalue())

        if st.button("Start Processing"):
            with st.spinner("Processing video..."):
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
                data_path = os.path.join(video_session_dir, "analysis_data.csv")

                try:
                    processing_path, _ = prepare_video_format(
                        temp_path, video_file.name
                    )

                    video_processor = VideoProcessor(model_config.segmentation_model)
                    _, _ = video_processor.process_video(
                        processing_path, result_path, data_path
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
                        "output_name": result_filename,
                        "session_dir": video_session_dir,
                    }
                    st.session_state.current_video = video_file.name

                    st.success(
                        f"Processing complete! Results saved to: {video_session_dir}"
                    )

                    if processing_path != temp_path:
                        os.unlink(processing_path)

                except Exception as error:
                    st.error(f"Processing error: {error}")
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
                st.info(f"Video saved: {current_results['video_output']}")

        with right_column:
            if os.path.exists(current_results["data_file"]):
                data_frame = pd.read_csv(current_results["data_file"])
                st.dataframe(data_frame.head(10))

                bubble_stats, overall_stats = analyze_bubble_statistics(
                    current_results["data_file"]
                )

                if not bubble_stats.empty:
                    st.info("### Bubble Statistics")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total bubbles", overall_stats["total_bubbles"])
                        st.metric(
                            "Average lifetime",
                            f"{overall_stats['avg_lifetime']:.2f} sec",
                        )
                        st.metric(
                            "Average area",
                            f"{overall_stats['avg_area_all']:.1f} px²",
                        )
                    with col2:
                        st.metric(
                            "Max speed",
                            f"{overall_stats['max_speed_all']:.1f} px/sec",
                        )
                        st.metric(
                            "Average speed",
                            f"{overall_stats['avg_speed_all']:.1f} px/sec",
                        )
                        st.metric(
                            "Speed (non-zero)",
                            f"{overall_stats['avg_nonzero_speed_all']:.1f} px/sec",
                        )

                    if "avg_path_length" in overall_stats:
                        st.info("### Trajectory Statistics")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                "Avg path length",
                                f"{overall_stats['avg_path_length']:.1f} px",
                            )
                        with col2:
                            st.metric(
                                "Straightness",
                                f"{overall_stats['avg_straightness']:.3f}",
                            )

                    st.info("### Detailed Bubble Statistics")
                    st.dataframe(bubble_stats)

with batch_tab:
    """Batch video processing tab."""
    st.subheader("Batch Video Processing")

    st.write("### Results Folder Settings")

    col1, col2 = st.columns(2)
    with col1:
        project_name = st.text_input("Project name", "cavitation_experiment")
    with col2:
        concentration = st.text_input("Solution concentration", "5%")

    batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    safe_concentration = concentration.replace("/", "_").replace("\\", "_")
    suggested_folder = f"{project_name}_{safe_concentration}_{batch_timestamp}"
    batch_folder_name = st.text_input("Results folder name", suggested_folder)

    multiple_videos = st.file_uploader(
        "Select video files for processing",
        type=["mp4", "avi", "mov"],
        accept_multiple_files=True,
    )

    if st.button("Start Batch Processing"):
        if multiple_videos and batch_folder_name:
            with st.spinner(f"Processing {len(multiple_videos)} videos..."):
                main_dir = os.path.join(BATCH_RESULTS_DIR, batch_folder_name)
                video_results_dir = os.path.join(main_dir, "processed_videos")
                data_results_dir = os.path.join(main_dir, "analysis_data")

                for directory in [video_results_dir, data_results_dir]:
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
                        _, _ = processor.process_video(
                            processing_path,
                            output_video_path,
                            output_data_path,
                        )

                        if os.path.exists(output_data_path):
                            video_data = pd.read_csv(output_data_path)
                            video_data["source_video"] = video.name
                            combined_data.append(video_data)

                        processing_results.append(
                            {
                                "video_name": video.name,
                                "output_video": output_video_path,
                                "data_file": "included in combined file",
                                "status": "Success",
                            }
                        )

                        if processing_path != temp_video_path:
                            os.unlink(processing_path)

                    except Exception as error:
                        processing_results.append(
                            {
                                "video_name": video.name,
                                "status": f"Error: {str(error)}",
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
                                bubble_stats,
                                overall_stats,
                                main_dir,
                                report_name,
                            )
                        )

                successful = len(
                    [r for r in processing_results if r["status"] == "Success"]
                )
                st.success(
                    f"Successfully processed: {successful} of {len(processing_results)}"
                )
                st.success(f"All results saved to: {main_dir}")

                results_table = pd.DataFrame(processing_results)
                st.dataframe(results_table)

                if combined_data and os.path.exists(combined_data_path):
                    st.success("### Bubble Statistics Analysis")

                    if not bubble_stats.empty:
                        st.info("### Detailed Bubble Statistics")
                        st.dataframe(bubble_stats)

                        st.info("### Overall Statistics")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total bubbles", overall_stats["total_bubbles"])
                            st.metric(
                                "Average lifetime",
                                f"{overall_stats['avg_lifetime']:.2f} sec",
                            )
                            st.metric(
                                "Lifetime std",
                                f"{overall_stats['std_lifetime']:.2f} sec",
                            )
                        with col2:
                            st.metric(
                                "Average area",
                                f"{overall_stats['avg_area_all']:.1f} px²",
                            )
                            st.metric(
                                "Max speed",
                                f"{overall_stats['max_speed_all']:.1f} px/sec",
                            )
                        with col3:
                            st.metric(
                                "Average speed",
                                f"{overall_stats['avg_speed_all']:.1f} px/sec",
                            )
                            st.metric(
                                "Speed (non-zero)",
                                f"{overall_stats['avg_nonzero_speed_all']:.1f} px/sec",
                            )
