import cv2
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

def adjust_contrast(image, contrast_value):
    """Return a contrast-adjusted copy of the image."""
    # contrast_value acts as alpha: >1 increases contrast, <1 decreases contrast
    return cv2.convertScaleAbs(image, alpha=contrast_value, beta=0)

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

def process_image_folder(image_dir, output_dir, contrast_values):
    """Process all image + XML pairs in a directory for multiple contrast values."""
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

            for value in contrast_values:
                # Adjust contrast
                contrast_img = adjust_contrast(image, value)

                # Save contrast image
                output_image_name = f"{base_name}_contrast_{value}.jpg"
                output_image_path = os.path.join(output_dir, output_image_name)
                cv2.imwrite(output_image_path, contrast_img)

                # Create updated XML
                output_xml_path = os.path.join(output_dir, f"{base_name}_contrast_{value}.xml")
                update_pascal_voc_xml(xml_path, output_image_name, output_image_path, contrast_img.shape, output_xml_path)

                print(f"✅ Processed: {file_name} → contrast {value}")

    print("\n🎉 All files processed successfully!")

# === Example Usage ===
if __name__ == "__main__":
    image_dir = r"C:\Users\deyro\OneDrive\Desktop\Video\train"  # Input directory
    output_dir = r"C:\Users\deyro\OneDrive\Desktop\Video\Dataset\Contrast_Output"  # Output directory
    brightness_values = [-5.0, -2.5,2.5, 5.0]

    process_image_folder(image_dir, output_dir, brightness_values)
