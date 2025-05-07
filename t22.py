import cv2
import math
import tkinter as tk
from tkinter import filedialog
import os
import sys
import numpy as np
from ultralytics import YOLO

# Load trained YOLO model
model_path = '/Users/vrajalpeshkumarmodi/Downloads/Cricket/YOLO/runs/detect/train/weights/best.pt'
model = YOLO(model_path)

# Open file dialog to select a video
root = tk.Tk()
root.withdraw()
video_path = filedialog.askopenfilename(title="Select a video file", filetypes=[("MP4 Files", "*.mp4"), ("All Files", "*.*")])

if not video_path:
    print("No video file selected.")
    exit()

# Setup output paths
code_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
video_filename = os.path.splitext(os.path.basename(video_path))[0]

desktop_path = os.path.expanduser("~/Desktop")
assignment_folder = os.path.join(desktop_path, "Vraj_Assignment")
os.makedirs(assignment_folder, exist_ok=True)

output_dir = os.path.join(assignment_folder, f"{video_filename}_{code_name}")
os.makedirs(output_dir, exist_ok=True)

output_video_path = os.path.join(output_dir, f'{video_filename}_{code_name}_tracking.mp4')

# Open video capture
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    print("Error: Unable to detect FPS.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
time_per_frame = 1 / fps

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

ball_path = []
bowler_path = []
prev_ball_center = None
max_ball_speed = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video processing completed.")
        break

    results = model.predict(source=frame, conf=0.5, save=False, verbose=False)

    if len(results) > 0:
        boxes = results[0].boxes.xywh.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        for box, cls in zip(boxes, classes):
            x_center, y_center, w, h = box
            cls = int(cls)

            x1, y1 = int(x_center - w / 2), int(y_center - h / 2)
            x2, y2 = int(x_center + w / 2), int(y_center + h / 2)

            if cls == 1:  # Ball
                current_center = (int(x_center), int(y_center))
                ball_path.append(current_center)

                if prev_ball_center is not None:
                    distance = math.hypot(current_center[0] - prev_ball_center[0],
                                          current_center[1] - prev_ball_center[1])
                    speed = distance / time_per_frame
                    max_ball_speed = max(max_ball_speed, speed)

                    cv2.putText(frame, f"Ball Speed: {speed:.2f} px/sec", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)

                prev_ball_center = current_center
                cv2.rectangle(frame, (x1, y1), (x2, y2), (35, 35, 164), 4)

            elif cls == 3:  # Bowler
                current_bowler_center = (int(x_center), int(y_center))
                bowler_path.append(current_bowler_center)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 4)

    # Draw smooth ball path (green)
    if len(ball_path) > 2:
        pts = np.array(ball_path, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=False, color=(0, 255, 0), thickness=4, lineType=cv2.LINE_AA)

    # Draw smooth bowler path (blue)
    if len(bowler_path) > 2:
        pts = np.array(bowler_path, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=False, color=(255, 0, 0), thickness=4, lineType=cv2.LINE_AA)

    # Display max ball speed
    cv2.putText(frame, f"Max Ball Speed: {max_ball_speed:.2f} px/sec", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 4)

    # Show the frame
    cv2.imshow('Tracking', frame)
    cv2.setWindowProperty('Tracking', cv2.WND_PROP_TOPMOST, 1)
    out.write(frame)

    if cv2.getWindowProperty('Tracking', cv2.WND_PROP_VISIBLE) < 1 or cv2.waitKey(1) & 0xFF == ord('q'):
        print("Video window was closed.")
        break


cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Tracking complete. Output saved at: {output_video_path}")
