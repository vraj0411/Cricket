import cv2
import os
from tkinter import Tk, filedialog
from ultralytics import YOLO
from datetime import datetime
from collections import Counter

# Class_id to name mapping
class_names = {
    0: "Drive",
    1: "Defensive",
    2: "Aggressive",
    3: "Leave"
}

# Colors for bar chart (BGR)
bar_colors = {
    0: (255, 0, 0),
    1: (0, 255, 0),
    2: (0, 0, 255),
    3: (255, 255, 0)
}

# Load YOLO model
model = YOLO("/Users/vrajalpeshkumarmodi/Downloads/Cricket/YOLO_2/runs/detect/train/weights/best.pt")


root = Tk()
root.withdraw()
video_path = filedialog.askopenfilename(title="Select a video file")

desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
output_folder = os.path.join(desktop_path, "Vraj_assignment")
os.makedirs(output_folder, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_video_path = os.path.join(output_folder, f"{os.path.basename(video_path).split('.')[0]}_output_{timestamp}.mp4")

cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

shot_counter = Counter()

def draw_bar_chart_on_frame(frame, counter):
    chart_x = 20
    chart_y = 50
    bar_width = 30
    max_bar_height = 150
    spacing = 100

    total = sum(counter.values())
    percentages = {i: (counter[i] / total * 100) if total > 0 else 0 for i in range(4)}

    for i in range(4):
        label = class_names[i]
        percent = percentages[i]
        bar_height = int((percent / 100) * max_bar_height)
        x1 = chart_x + i * spacing
        y1 = chart_y + max_bar_height
        x2 = x1 + bar_width
        y2 = y1 - bar_height

        # Draw the bar
        cv2.rectangle(frame, (x1, y1), (x2, y2), bar_colors[i], -1)

        # Draw the label and percentage
        cv2.putText(frame, label, (x1 - 10, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bar_colors[i], 2)
        cv2.putText(frame, f"{percent:.1f}%", (x1, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)

# Frame-by-frame processing
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = f"{class_names.get(cls_id, 'Unknown')} ({conf:.2f})"

        # Draw detection
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 4)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 4)

        
        shot_counter[cls_id] += 1

    # Draw the live bar chart on the video frame
    draw_bar_chart_on_frame(frame, shot_counter)

    # Show and save frame
    out.write(frame)
    cv2.imshow("Shot Detection with Live Chart", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video saved at: {output_video_path}")
