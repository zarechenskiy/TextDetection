import os
from xml.etree import cElementTree as ET


def convert_file(filename, groundtruth):
    with open(filename, 'r') as f:
        tag_set = ET.fromstring(f.read())
        for image in list(tag_set):
            image_path = image.find("imageName").text
            folder, img_name = image_path.split("/")
            rectangles = image.find("taggedRectangles")

            groundtruth_folder = groundtruth + "/" + folder
            if not os.path.exists(groundtruth_folder):
                os.makedirs(groundtruth_folder)

            with open(groundtruth_folder + "/" + img_name.split('.')[0] + ".txt", 'w') as res:
                for rect in rectangles:
                    x = rect.get('x')
                    y = rect.get('y')
                    xx = str(float(x) + float(rect.get("width")))
                    yy = str(float(y) + float(rect.get("height")))
                    res.write(x + " " + y + " " + xx + " " + yy + os.linesep)


convert_file("../datasets/icdar2003_test/locations.xml", "groundtruth")