import os
import shutil

# Path to your folder
FOLDER = r"train"

# Allowed image formats
IMAGE_EXTS = [".jpg", ".jpeg", ".png"]

# Collect image-xml pairs
pairs = []
for file in os.listdir(FOLDER):
    filepath = os.path.join(FOLDER, file)
    name, ext = os.path.splitext(file)

    if ext.lower() in IMAGE_EXTS:
        xml_path = os.path.join(FOLDER, name + ".xml")
        if os.path.exists(xml_path):
            pairs.append((filepath, xml_path))

# Step 2 — sort for consistent numbering
pairs.sort()

# Step 3 — rename all pairs using only numbers
i = 1
for img_path, xml_path in pairs:
    _, img_ext = os.path.splitext(img_path)

    new_img = os.path.join(FOLDER, f"{i}{img_ext.lower()}")
    new_xml = os.path.join(FOLDER, f"{i}.xml")

    shutil.move(img_path, new_img)
    shutil.move(xml_path, new_xml)

    i += 1

print("DONE — All files renamed using numbers only. Duplicates kept!")
