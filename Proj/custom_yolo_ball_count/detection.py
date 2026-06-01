import ultralytics
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO


model = YOLO('yolov8s.pt')
ultralytics.checks()

folder_name = "output"
#os.mkdir(folder_name)


input_image_folder = 'input/'

# Predict bounding boxes
results = model.predict(input_image_folder)

for result in results:

    boxes = result.boxes

    image_name = result.path.split(os.sep)[-1]
    result.save(filename=os.path.join(folder_name, f"yolo_{image_name}"))
