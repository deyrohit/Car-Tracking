import cv2
import os
from tqdm import tqdm

# === CONFIG ===
video_path = r"C:\Users\deyro\OneDrive\Desktop\Video\Video.mp4"  # path to your video file
output_folder = "Frames"  
frame_skip = 5 

# === SETUP ===
os.makedirs(output_folder, exist_ok=True)
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Error: Cannot open video file.")
    exit()

# Get total frame count for progress bar
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frame_count = 0
saved_count = 0

# === PROCESS WITH PROGRESS BAR ===
with tqdm(total=total_frames, desc="Extracting frames", unit="frame") as pbar:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1
        pbar.update(1)

cap.release()
print(f"\n✅ Done! Extracted {saved_count} frames to: {output_folder}")
