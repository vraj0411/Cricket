import tkinter as tk
from tkinter import filedialog
import subprocess
import os
import sys


root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(title="Select a video file", filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")])

if file_path:
   
    code_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]  

    #output folder on Desktop 
    desktop_path = os.path.expanduser("~/Desktop")
    output_folder = os.path.join(desktop_path, "Vraj_Assignment")
    os.makedirs(output_folder, exist_ok=True)  # Create if not exists

    
    input_filename = os.path.splitext(os.path.basename(file_path))[0]


    save_dir = os.path.join(output_folder, f"{input_filename}_{code_name}")

    command = (
        f"yolo detect predict model=runs/detect/train/weights/best.pt "
        f"source=\"{file_path}\" "
        f"save=True "
        f"save_txt=True "
        f"project=\"{output_folder}\" "
        f"name=\"{input_filename}_{code_name}\" "
        f"exist_ok=True"
    )

    subprocess.run(command, shell=True)
    print(f"Predictions saved at: {save_dir}")

else:
    print("No file selected.")
