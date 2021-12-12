import os
import xml.etree.ElementTree as ET

input_file_path = 'annotations.xml'
output_folder_path = 'ground_truth'

tree = ET.parse(input_file_path)
root = tree.getroot()

classes_mapping_dict = {
    "person": 0
}

for image in root.iter('image'):
    frame_name = image.attrib['name']
    with open(os.path.join(output_folder_path, frame_name+'.txt'), "w") as out:
        for box in image.iter('box'):
            label = str(box.attrib['label'])
            class_id, tracking_id = label.split("_")
            class_id = classes_mapping_dict[class_id]
            confidence = 1
            visibility = 1
            xtl = int(float(box.attrib['xtl']))
            ytl = int(float(box.attrib['ytl']))
            xbr = int(float(box.attrib['xbr']))
            ybr = int(float(box.attrib['ybr']))
            w = xbr - xtl
            h = ybr - ytl
            out.write(f"{frame_name}, {tracking_id}, {xtl}, {ytl}, {w}, {h}, {confidence}, {class_id}, {visibility}\n")

