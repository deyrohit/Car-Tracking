import os
import cv2
import xml.etree.ElementTree as ET
from xml.dom import minidom
from ultralytics import YOLO
from tqdm import tqdm
import shutil

# === CONFIG ===
MODEL_PATH = r"C:\Users\deyro\OneDrive\Desktop\Video\epoches\train\weights\best.pt"      # path to your YOLO model
IMAGE_DIR = r"C:\Users\deyro\OneDrive\Desktop\Video\Frames"         # input folder containing images
OUTPUT_DIR = "Output"        # output folder for images + XML
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === LOAD YOLO MODEL ===
model = YOLO(MODEL_PATH)

# === Helper: Create Pascal VOC XML ===
def create_voc_xml(image_path, output_path, detections, class_names):
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Skipping invalid image: {image_path}")
        return
    h, w, c = img.shape

    annotation = ET.Element("annotation")

    ET.SubElement(annotation, "folder").text = os.path.basename(os.path.dirname(image_path))
    ET.SubElement(annotation, "filename").text = os.path.basename(image_path)
    ET.SubElement(annotation, "path").text = os.path.abspath(image_path)

    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "Unknown"

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(w)
    ET.SubElement(size, "height").text = str(h)
    ET.SubElement(size, "depth").text = str(c)

    ET.SubElement(annotation, "segmented").text = "0"

    for det in detections:
        class_id = int(det[5])
        label = class_names[class_id]
        xmin, ymin, xmax, ymax = map(int, det[:4])

        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = label
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"

        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(xmin)
        ET.SubElement(bbox, "ymin").text = str(ymin)
        ET.SubElement(bbox, "xmax").text = str(xmax)
        ET.SubElement(bbox, "ymax").text = str(ymax)

    # Pretty-print XML
    xml_str = minidom.parseString(ET.tostring(annotation)).toprettyxml(indent="   ")
    with open(output_path, "w") as f:
        f.write(xml_str)

# === MAIN ===
image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

for img_name in tqdm(image_files, desc="Generating XML annotations"):
    img_path = os.path.join(IMAGE_DIR, img_name)
    output_image_path = os.path.join(OUTPUT_DIR, img_name)

    # Copy original image to output
    shutil.copy(img_path, output_image_path)

    # Run YOLO detection
    results = model(img_path, verbose=False)[0]

    detections = []
    for box in results.boxes.data.tolist():  # each box = [x1, y1, x2, y2, conf, class]
        detections.append(box)

    # Create corresponding XML file
    xml_filename = os.path.splitext(img_name)[0] + ".xml"
    xml_output_path = os.path.join(OUTPUT_DIR, xml_filename)

    create_voc_xml(img_path, xml_output_path, detections, results.names)

print(f"\n✅ Done! Original images + XML saved in: {OUTPUT_DIR}")
