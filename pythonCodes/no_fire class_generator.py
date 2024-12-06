import os
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString

# Path to the folder containing image files
folder_path = "my"

# Parameters for the XML template
width = 640
height = 640
depth = 3
label = "no fire"

# Create XML for each image
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg"):  # Process only JPG files
        base_name = os.path.splitext(filename)[0]
        file_path = os.path.join(folder_path, filename)

        # Create the XML structure
        annotation = Element("annotation")

        folder = SubElement(annotation, "folder")
        folder.text = ""

        file_name = SubElement(annotation, "filename")
        file_name.text = filename

        path = SubElement(annotation, "path")
        path.text = filename

        source = SubElement(annotation, "source")
        database = SubElement(source, "database")
        database.text = "roboflow.com"

        size = SubElement(annotation, "size")
        width_elem = SubElement(size, "width")
        width_elem.text = str(width)
        height_elem = SubElement(size, "height")
        height_elem.text = str(height)
        depth_elem = SubElement(size, "depth")
        depth_elem.text = str(depth)

        segmented = SubElement(annotation, "segmented")
        segmented.text = "0"

        # Add the label for the image
        obj = SubElement(annotation, "object")
        name = SubElement(obj, "name")
        name.text = label
        pose = SubElement(obj, "pose")
        pose.text = "Unspecified"
        truncated = SubElement(obj, "truncated")
        truncated.text = "0"
        difficult = SubElement(obj, "difficult")
        difficult.text = "0"
        occluded = SubElement(obj, "occluded")
        occluded.text = "0"

        # Beautify and write XML to file
        xml_str = parseString(tostring(annotation)).toprettyxml(indent="    ")
        xml_path = os.path.join(folder_path, f"{base_name}.xml")
        with open(xml_path, "w") as xml_file:
            xml_file.write(xml_str)

print(f"XML files with labels for 'no fire' have been generated in the '{folder_path}' directory.")
