import cv2
import os
from tkinter import Tk, filedialog
from ultralytics import YOLO

model_path = '/Users/vrajalpeshkumarmodi/Downloads/Cricket/YOLO/runs/detect/train/weights/best.pt' 
confidence_threshold = 0.5  

ball_class_id = 1  # Ball class ID
bat_class_id = 0   # Bat class ID


Tk().withdraw() 
video_path = filedialog.askopenfilename(title="Select a Video File",
                                        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")])
if not video_path:
    print("No video selected. Exiting...")
    exit()

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Selected Video: {video_path}")
print(f"FPS Detected: {fps}")

#Prepare output folder and filename 
input_filename = os.path.splitext(os.path.basename(video_path))[0]
code_name = "counterfactual"
output_filename = f"{input_filename}_{code_name}.mp4"

desktop_path = os.path.expanduser("~/Desktop")
assignment_folder = os.path.join(desktop_path, "Vraj_Assignment")
os.makedirs(assignment_folder, exist_ok=True)  # Create if not exists

output_video_path = os.path.join(assignment_folder, output_filename)

print(f"Output will be saved at: {output_video_path}")

model = YOLO(model_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

ball_path_real = []
counterfactuals = {
    'fast': [],
    'faster': [],
    'slow': [],
    'slower': []
}

speed_factors = {
    'fast': 1.10,
    'faster': 1.20,
    'slow': 0.90,
    'slower': 0.80
}

frame_idx = 0
paused = False

while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects using YOLO
        results = model.predict(source=frame, conf=confidence_threshold, save=False, verbose=False)

        if len(results) > 0:
            boxes = results[0].boxes.xywh.cpu().numpy()  # [x_center, y_center, width, height]
            classes = results[0].boxes.cls.cpu().numpy() # Class IDs

            for box, cls in zip(boxes, classes):
                x_center, y_center, w, h = box
                cls = int(cls)

                x1 = int(x_center - w / 2)
                y1 = int(y_center - h / 2)
                x2 = int(x_center + w / 2)
                y2 = int(y_center + h / 2)

                # Ball detection 
                if cls == ball_class_id:
                    current_center = (int(x_center), int(y_center))

                    # Save ball center path
                    ball_path_real.append(current_center)

                    # Draw bounding box on ball (Green)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                 
                    if len(ball_path_real) >= 2:
                        prev_real = ball_path_real[-2]
                        dx = current_center[0] - prev_real[0]
                        dy = current_center[1] - prev_real[1]

                        for key, factor in speed_factors.items():
                            if len(counterfactuals[key]) == 0:
                                counterfactuals[key].append(prev_real)

                            last_point = counterfactuals[key][-1]
                            new_point = (int(last_point[0] + dx * factor), int(last_point[1] + dy * factor))
                            counterfactuals[key].append(new_point)

        
                elif cls == bat_class_id:
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

 
        for i in range(1, len(ball_path_real)):
            cv2.line(frame, ball_path_real[i-1], ball_path_real[i], (0, 255, 0), 4)

        # Counterfactual Paths
        colors = {
            'fast': (0, 0, 255),    # Red
            'faster': (255, 0, 0),  # Blue
            'slow': (0, 255, 255),  # Yellow
            'slower': (255, 0, 255) # Pink
        }

        for key, path in counterfactuals.items():
            for i in range(1, len(path)):
                cv2.line(frame, path[i-1], path[i], colors[key], 2)

      
        cv2.putText(frame, "Real Path (Green)", (30, height - 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 4)
        cv2.putText(frame, "Fast (Red)", (30, height - 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 4)
        cv2.putText(frame, "Faster (Blue)", (30, height - 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 4)
        cv2.putText(frame, "Slow (Yellow)", (30, height - 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 4)
        cv2.putText(frame, "Slower (Pink)", (30, height - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 4)

    
        out.write(frame)

       
        cv2.imshow('Counterfactual Simulation', frame)

   
    key = cv2.waitKey(1) & 0xFF
    if key == ord('p'):
        paused = not paused  
        if paused:
            print("Video Paused. Press 'p' again to resume.")
        else:
            print("Video Resumed.")
    elif key == ord('q'):
        print("🚪 Exiting video...")
        break

    frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Final video saved inside 'Vraj_Assignment' at: {output_video_path}")
