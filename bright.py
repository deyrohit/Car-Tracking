import cv2
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

def adjust_brightness(image, brightness_value):
    """Return a brightness-adjusted copy of the image."""
    return cv2.convertScaleAbs(image, alpha=1, beta=brightness_value)

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

def process_image_folder(image_dir, output_dir, brightness_values):
    """Process all image + XML pairs in a directory for multiple brightness values."""
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

            for value in brightness_values:
                # Brightness adjustment
                bright_img = adjust_brightness(image, value)

                # Save bright image
                output_image_name = f"{base_name}_bright_{value}.jpg"
                output_image_path = os.path.join(output_dir, output_image_name)
                cv2.imwrite(output_image_path, bright_img)

                # Create updated XML
                output_xml_path = os.path.join(output_dir, f"{base_name}_bright_{value}.xml")
                update_pascal_voc_xml(xml_path, output_image_name, output_image_path, bright_img.shape, output_xml_path)

                print(f"✅ Processed: {file_name} → brightness {value}")

    print("\n🎉 All files processed successfully!")

# === Example Usage ===
if __name__ == "__main__":
    image_dir = r"C:\Users\deyro\OneDrive\Desktop\Video\train"  # Input directory
    output_dir = r"C:\Users\deyro\OneDrive\Desktop\Video\Dataset\Bright_Output"  # Output directory
    brightness_values = [
    -50.0, -47.5, -45.0, -42.5, -40.0, -37.5, -35.0, -32.5, -30.0, -27.5,
    -25.0, -22.5, -20.0, -17.5, -15.0, -12.5, -10.0, -7.5, -5.0, -2.5,
    2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0,
    27.5, 30.0, 32.5, 35.0, 37.5, 40.0, 42.5, 45.0, 47.5, 50.0
]

    process_image_folder(image_dir, output_dir, brightness_values)
