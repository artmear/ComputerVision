import numpy as np
import cv2
import os

def visualize_table_mask_debug(image, image_name, output_dir, table_mask):
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(image_name)[0]
    debug_path = os.path.join(output_dir, f"{base_name}_table_mask.png")

    cv2.imwrite(debug_path, table_mask)
    print(f"[DEBUG] table mask saved in: {debug_path}")

def save_debug_image(image, image_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(image_name)[0]
    debug_path = os.path.join(output_dir, f"{base_name}_debug.png")

    cv2.imwrite(debug_path, image)
    print(f"[DEBUG] image saved in: {debug_path}")