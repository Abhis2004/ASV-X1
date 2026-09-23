import cv2
import numpy as np
import os
import sys
from ultralytics import YOLO

# Add current directory to path for tracker import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tracker import DroneTracker

# Video & Model Config
video_path = r"C:\Users\araj8\Music\test_data\following.mp4"
model_path = r"model\best.pt"

if not model_path or not os.path.exists(model_path):
    model_path = "yolov8s.pt"

model = YOLO(model_path)

selected_id = None
current_boxes = []


def mouse_callback(event, x, y, flags, param):
    global selected_id

    if event == cv2.EVENT_LBUTTONDOWN:
        for box, track_id in current_boxes:
            x1, y1, x2, y2 = box

            if x1 <= x <= x2 and y1 <= y <= y2:
                selected_id = track_id
                print(f"[TARGET LOCK] Selected Track ID: {selected_id}")
                break


def draw_hud(frame, selected_id, track_count):
    """Draw professional drone telemetry HUD header."""
    h, w, _ = frame.shape

    # Semi-transparent header panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Telemetry text
    status_text = f"DRONE TRACKER v2.0 | ACTIVE TARGETS: {track_count}"
    cv2.putText(frame, status_text, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    if selected_id is not None:
        target_text = f"LOCK-ON: ID #{selected_id}"
        cv2.putText(frame, target_text, (w - 200, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        target_text = "LOCK-ON: NONE (CLICK TO SELECT)"
        cv2.putText(frame, target_text, (w - 280, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1, cv2.LINE_AA)


def draw_target_reticle(frame, x1, y1, x2, y2, color=(0, 0, 255)):
    """Draw tactical lock-on reticle corners around selected target."""
    length = max(8, int(min(x2 - x1, y2 - y1) * 0.25))

    # Top-Left corner
    cv2.line(frame, (x1, y1), (x1 + length, y1), color, 2)
    cv2.line(frame, (x1, y1), (x1, y1 + length), color, 2)

    # Top-Right corner
    cv2.line(frame, (x2, y1), (x2 - length, y1), color, 2)
    cv2.line(frame, (x2, y1), (x2, y1 + length), color, 2)

    # Bottom-Left corner
    cv2.line(frame, (x1, y2), (x1 + length, y2), color, 2)
    cv2.line(frame, (x1, y2), (x1, y2 - length), color, 2)

    # Bottom-Right corner
    cv2.line(frame, (x2, y2), (x2 - length, y2), color, 2)
    cv2.line(frame, (x2, y2), (x2, y2 - length), color, 2)


def track(video_path):
    global current_boxes, selected_id

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file: {video_path}")
        return

    # Initialize professional drone tracker
    tracker = DroneTracker(
        det_thresh=0.20,     # High confidence threshold
        low_thresh=0.05,     # Low confidence threshold for ByteTrack 2nd stage
        match_thresh=0.80,   # DIoU matching threshold
        max_age=35,          # Retain lost tracks for 35 frames during occlusions
        min_hits=2,          # Require 2 consecutive detections to confirm track
        use_gmc=True         # Enable Camera Motion Compensation for drone motion
    )

    window_name = "ASV-X1 Professional Drone Tracking (VisDrone)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("[INFO] Drone tracking started. Press 'q' to quit, click on any target box to lock-on.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (960, 540))

        # Run YOLO detection
        results = model.predict(frame, conf=0.05, imgsz=640, verbose=False)
        result = results[0]

        # Extract raw detections [x1, y1, x2, y2, conf, cls]
        detections = []
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for box, conf, cls in zip(boxes, confs, classes):
                x1, y1, x2, y2 = box
                detections.append([x1, y1, x2, y2, conf, cls])

        # Update standalone drone tracker
        active_tracks = tracker.update(detections, frame)

        current_boxes = []

        for track_obj in active_tracks:
            track_id = track_obj.track_id
            x1, y1, x2, y2 = map(int, track_obj.to_tlbr())
            cls_id = track_obj.cls
            conf = track_obj.conf

            current_boxes.append(((x1, y1, x2, y2), track_id))

            is_selected = (track_id == selected_id)
            color = (0, 0, 255) if is_selected else (0, 255, 0)
            thickness = 2 if is_selected else 1

            # 1. Draw trajectory trail
            if len(track_obj.history) > 1:
                pts = np.array(track_obj.history, dtype=np.int32).reshape((-1, 1, 2))
                trail_color = (50, 50, 255) if is_selected else (0, 200, 100)
                cv2.polylines(frame, [pts], False, trail_color, thickness=2 if is_selected else 1)

            # 2. Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # 3. Draw lock-on reticle if target selected
            if is_selected:
                draw_target_reticle(frame, x1, y1, x2, y2, color=(0, 0, 255))

            # 4. Label tag
            class_name = model.names.get(cls_id, f"C:{cls_id}")
            label = f"ID:{track_id} | {class_name} {conf:.2f}"

            # Label text background
            (txt_w, txt_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, y1 - txt_h - 6), (x1 + txt_w + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw HUD header
        draw_hud(frame, selected_id, len(active_tracks))

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    track(video_path)
