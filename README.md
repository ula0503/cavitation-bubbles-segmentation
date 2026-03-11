# Cavitation Bubbles Segmentation

Segmentation and tracking of cavitation bubbles using YOLO and ByteTrack.

## What it does
- Detects bubbles in video using YOLOv11m-seg
- Tracks each bubble with ByteTrack + Kalman filter
- Calculates area, speed, trajectory, lifetime
- Saves annotated videos and CSV with all data

## Training
- 335 images labeled in Roboflow (in-focus / out-of-focus)
- 150 epochs, YOLOv11m-seg
- Final mAP50: 0.895

## Output parameters
- area (px²)
- speed (px/sec)
- displacement (movement between frames)
- total_displacement (start to end point)
- path_length (total distance)
- confidence
- lifetime (sec)

  
## Files
- `improved_streamlit.py` — main app
- `src/` — all processing code
- `models/` — trained models
