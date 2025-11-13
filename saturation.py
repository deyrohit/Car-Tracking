import cv2
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
import numpy as np

def adjust_saturation(image, saturation_factor):
    """Return a saturation-adjusted copy of the image."""
    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype("float32")

    # Scale the S (saturation) channel
    h, s, v = cv2.split(hsv)
    s = np.clip(s * saturation_factor, 0, 255)
    hsv = cv2.merge([h, s, v])

    # Convert back to BGR
    return cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)

def update_pascal_voc_xml(xml_path, output_image_name, output_image_path, image_shape, output_xml_path):
    """Read an existing Pascal VOC XML and update filename/path."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Update filename and path
    filename_tag = root.find("filename")
    if filename_tag is not None:
        filename_tag.text = output_image_name

    path_tag = root.find("path")
    if path_tag is not None:
        path_tag.text = output_image_path
    else:
        ET.SubElement(root, "path").text = output_image_path

    # Update size
    size_tag = root.find("size")
    if size_tag is not None:
        height, width, depth = image_shape
        size_tag.find("width").text = str(width)
        size_tag.find("height").text = str(height)
        size_tag.find("depth").text = str(depth)

    # Pretty print XML and save
    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
    with open(output_xml_path, "w") as f:
        f.write(xml_str)

def process_image_folder(image_dir, output_dir, saturation_values):
    """Process all image + XML pairs in a directory for multiple saturation levels."""
    os.makedirs(output_dir, exist_ok=True)
    valid_exts = [".jpg", ".jpeg", ".png"]

    for file_name in os.listdir(image_dir):
        if any(file_name.lower().endswith(ext) for ext in valid_exts):
            image_path = os.path.join(image_dir, file_name)
            xml_path = os.path.splitext(image_path)[0] + ".xml"

            if not os.path.exists(xml_path):
                print(f"⚠️ Skipping {file_name} — XML not found.")
                continue

            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Cannot read image: {file_name}")
                continue

            base_name = os.path.splitext(file_name)[0]

            for factor in saturation_values:
                # Adjust saturation
                sat_img = adjust_saturation(image, factor)

                # Save saturated image
                output_image_name = f"{base_name}_saturation_{factor}.jpg"
                output_image_path = os.path.join(output_dir, output_image_name)
                cv2.imwrite(output_image_path, sat_img)

                # Create updated XML
                output_xml_path = os.path.join(output_dir, f"{base_name}_saturation_{factor}.xml")
                update_pascal_voc_xml(xml_path, output_image_name, output_image_path, sat_img.shape, output_xml_path)

                print(f"✅ Processed: {file_name} → saturation ×{factor}")

    print("\n🎉 All files processed successfully!")

# === Example Usage ===
if __name__ == "__main__":
    image_dir = r"C:\Users\deyro\OneDrive\Desktop\Video\train"  # Input directory
    output_dir = r"C:\Users\deyro\OneDrive\Desktop\Video\Dataset\Saturation_Output"  # Output directory
    brightness_values = [ 2.5, 5.0]
    process_image_folder(image_dir, output_dir, brightness_values)
