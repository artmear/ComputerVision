import numpy as np
import cv2
import os
import json

from ball_detection import detect_balls
from warp_table import warp_table

json_input = "example_json/input.json"

def read_from_json(input_dir):
    with open(input_dir) as json_data:
        return json.load(json_data)["image_path"]
 
def write_to_json(output_path, contents):
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(contents, file, ensure_ascii=False, indent=2)
 
def read_image(images, image_index):
    full_path = images[image_index]
 
    img_pool_table = cv2.imread(full_path)
 
    image_name = os.path.basename(full_path)
 
    return img_pool_table, image_name


images = read_from_json(json_input)
for i in range(len(images)):
    table_image, img_name = read_image(images, i)

    #warp_table(table_image, img_name, output_dir="output")
    detect_balls(table_image, img_name, output_dir="output", debugOutput=False)
    warp_table(table_image, img_name, output_dir="output")