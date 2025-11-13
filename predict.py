import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
import re

# Load YOLOv8 model (custom trained with "Number Plate" label)
yolo = YOLO(r'C:\Users\deyro\OneDrive\Desktop\Video\best.pt')  # Replace with your fine-tuned model path if needed

# Initialize PaddleOCR with specified options
ocr = PaddleOCR(
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

def extract_distinct_number_plates(video_path):
    cap = cv2.VideoCapture(video_path)
    number_plate_class = "Number Plate"
    detected_plates = set()
    
    plate_pattern = re.compile(r'^[A-Za-z0-9]{7,}$')   
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = yolo(frame, conf=0.5)
        
        for box, cls_id in zip(results[0].boxes, results[0].cls):
            class_idx = int(cls_id)
            if yolo.names[class_idx] != number_plate_class:
                continue
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_img = frame[y1:y2, x1:x2]
            if plate_img.size == 0:
                continue
            
            temp_file = "temp_plate.jpg"
            cv2.imwrite(temp_file, plate_img)
            
            ocr_results = ocr.predict(temp_file)
            for res in ocr_results:
                for line in res.result:
                    text = line[1][0].strip().replace(" ", "")  # remove spaces
                    
                    if plate_pattern.match(text):
                        detected_plates.add(text)
    
    cap.release()
    return detected_plates

if __name__ == "__main__":
    video_file = r"C:\Users\deyro\OneDrive\Desktop\Video\Video.mp4"
    unique_plates = extract_distinct_number_plates(video_file)
    
    print("Distinct Number Plates detected in video:")
    for plate in unique_plates:
        print(plate)
