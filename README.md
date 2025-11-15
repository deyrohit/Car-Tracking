# 🚗 Car Real-Time Tracking System

A real-time vehicle tracking pipeline built using **YOLO** for detection and a **custom centroid-based tracker** for consistent multi-object identification, event logging, and annotated video generation.

---

## 🔥 Key Features

- **Real-time vehicle detection** using YOLO.  
- **Centroid-based multi-object tracking** to maintain unique IDs across frames.  
- **< 50 px matching distance** for accurate object association.  
- **Entry/Exit event logging** with timestamps.  
- **Per-vehicle duration tracking** inside the monitored zone.  
- **Annotated output video generation** with bounding boxes, IDs, and tracking info.  
- **Robust performance** against frame drops, minor occlusions, and ID switching.  

---

## 🛠️ What the System Does

- Detects cars frame-by-frame in real time.  
- Computes centroids and matches them with previous frame centroids.  
- Assigns and maintains **unique IDs** for each vehicle.  
- Tracks each vehicle’s path using centroid motion.  
- Logs:  
  - Vehicle entry time  
  - Vehicle exit time  
  - Total duration inside the region  
- Produces visually annotated video with:  
  - Bounding boxes  
  - Vehicle IDs  
  - Tracking lines  
  - Timestamp overlays  

---

## 📌 Technical Highlights

- High-performance processing loop optimized for real-time speed.  
- Accurate multi-object tracking with centroid distance threshold (<50 px).  
- Handles scenarios of partial occlusion and re-identification.  
- Maintains a dictionary of active and completed tracks for logging.  
- Writes logs to CSV / text files.  

---

## 🧰 Tech Stack

- **YOLO (You Only Look Once)** – Vehicle detection  
- **OpenCV** – Video processing & drawing overlays  
- **Python** – Core implementation  
- **NumPy** – Coordinate + distance calculations  

---

## 📁 Output Examples

- **Annotated video** showing all tracked vehicles with IDs  
- **Log file** containing entry/exit timestamps and durations  
