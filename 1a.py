import cv2
from ultralytics import YOLO
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os

# YOLOv8 Model
model_path_1 = "/Users/vrajalpeshkumarmodi/Downloads/Cricket/Yolo/runs/detect/train/weights/best.pt"  
model_path_2 = "/Users/vrajalpeshkumarmodi/Downloads/Cricket/Yolo_3/runs/detect/train/weights/best.pt"  

model_1 = YOLO(model_path_1)
model_2 = YOLO(model_path_2)

view = input("Is the camera view front or rear? (f/r): ").strip().lower()
if view not in ['f', 'r']:
    print("Invalid input. Please enter 'f' or 'r'.")
    exit()

Tk().withdraw()
video_path = askopenfilename(title="Select a video", filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])

if not video_path:
    print("No video selected.")
    exit()

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error opening video.")
    exit()

desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
output_folder = os.path.join(desktop_path, "Vraj_Assignment")
os.makedirs(output_folder, exist_ok=True)

video_name = os.path.splitext(os.path.basename(video_path))[0]
output_path = os.path.join(output_folder, f"{video_name}_output.mp4")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print("Running inference...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    results_1 = model_1(frame)[0]
    results_2 = model_2(frame)[0]

    # Ball and Batsman Detection (Model 1) 
    for box in results_1.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if cls_id == 1 and conf > 0.5:  # Ball
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
            cv2.putText(frame, f"Ball {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 4)

        elif cls_id == 4 and conf > 0.5:  # Batsman
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 4)
            cv2.putText(frame, f"Batsman {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 4)


    for box in results_2.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if conf > 0.5:
            label = "Right Hand" if cls_id == 1 else "Left Hand"
            color = (0, 0, 255) if cls_id == 1 else (255, 0, 0)

            # Swap labels if rear view
            if view == 'r':
                label = "Left Hand" if cls_id == 1 else "Right Hand"
                color = (255, 0, 0) if cls_id == 1 else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 4)

    out.write(frame)
    cv2.imshow("YOLOv8 Video Inference", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Inference complete. Output saved to:\n{output_path}")
