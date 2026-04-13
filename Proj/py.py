import cv2
import os
import json

from ball_detection import detect_balls
from warp_table import warp_table
from number_detection import detect_ball_numbers, show_detected_balls_on_image

INPUT_DIR = "example_json/input.json" # TODO: Change this to the desired input json file
OUTPUT_DIR = "output" # TODO: Change this to the desired output directory
DEBUG_OUTPUT_DIR = "debug" # TODO: Change this to the desired debug output directory

DEBUG_IMG_NUM = 6
DEBUG_ALL = 1

def read_from_json(input_dir):
    with open(input_dir) as json_data:
        return json.load(json_data)["image_path"]
 
def write_to_json(output_path, contents):
    os.makedirs(os.path.dirname(output_path), exist_ok=True) # Create the output directory if it doesn't exist
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(contents, file, ensure_ascii=False, indent=4)
 
def read_image(images, image_index):
    full_path = images[image_index]
 
    img_pool_table = cv2.imread(full_path)
 
    image_name = os.path.basename(full_path)
 
    return img_pool_table, image_name


images = read_from_json(INPUT_DIR)

if not DEBUG_ALL:
    ##### Ball detection #####
    table_image, img_name = read_image(images, DEBUG_IMG_NUM)
    circles = detect_balls(table_image, img_name, output_dir=DEBUG_OUTPUT_DIR, debugOutput=False)

    ##### Number identification #####
    balls = detect_ball_numbers(table_image, circles, img_name)
    show_detected_balls_on_image(table_image, balls)


##### DEBUG All Loop ####
if DEBUG_ALL:
    for i in range(len(images)):    
        table_image, img_name = read_image(images, i)
        circles = detect_balls(table_image, img_name, output_dir=DEBUG_OUTPUT_DIR+"/top_view", debugOutput=False) # FIXME: The output_dir is for warp_table (not ball detection)
        balls = detect_ball_numbers(table_image, circles, img_name)
        warp_table(table_image, img_name, output_dir=DEBUG_OUTPUT_DIR+"/top_view") 


''' Entry point for the full detection pipeline
    - Reads the images from the paths in the input json file  
    - Detects the balls and their numbers
    - Warps the table to a top-down view
    - Saves the results to the output directory (in json and images)
'''
def run_full_detection(json_input_dir, output_dir):
    # 1. Read the image paths array from the json file
    image_paths = read_from_json(json_input_dir)

    # 2. Loop through each image path and run the detection pipeline
    i = 0
    output_data_list = []
    for image_path in image_paths:
        # 2.1 Read the image
        table_image, img_name = read_image(image_paths, i)

        # 2.2 Detect balls
        circles = detect_balls(table_image, img_name, output_dir=DEBUG_OUTPUT_DIR+"/top_view", debugOutput=False) # FIXME: The output_dir is for warp_table (not ball detection)

        # 2.3 Detect ball numbers
        balls = detect_ball_numbers(table_image, circles, img_name) 

        # 2.4 Warp the table and save the warped image
        warp_table(table_image, img_name, output_dir=output_dir+"/top_view") 

        # 2.5. Create JSON object with output data
        output_data = {
            "image_path": image_path,
            "num_balls": len(balls),
            "balls": [ball.to_dict() for ball in balls]
        }
        output_data_list.append(output_data)

        i += 1

    # 2.6 Save the results to the output directory (in json)
    output_json_path = os.path.join(output_dir, "detection_results.json")
    write_to_json(output_json_path, output_data_list)



run_full_detection(INPUT_DIR, OUTPUT_DIR)