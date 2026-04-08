import numpy as np
import cv2
import os
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Debug da deteção da mesa (original | filter | resultado)")
    parser.add_argument("--input-json", type=str, default="example_json/input.json")
    parser.add_argument("--max-images", type=int, default=15)
    parser.add_argument("--debug-dir", type=str, default="output_debug")
    parser.add_argument("--output-dir", type=str, default="output")
    return parser.parse_args()

def visualize_table_mask_debug(image, image_name, output_dir, filter_mask, result_image):
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(image_name)[0]
    debug_path = os.path.join(output_dir, f"{base_name}_debug.jpg")

    if len(filter_mask.shape) == 2:
        filter_panel = cv2.cvtColor(filter_mask, cv2.COLOR_GRAY2BGR)
    else:
        filter_panel = filter_mask.copy()

    h, w = image.shape[:2]
    filter_panel = cv2.resize(filter_panel, (w, h), interpolation=cv2.INTER_NEAREST)
    result_panel = cv2.resize(result_image, (w, h), interpolation=cv2.INTER_LINEAR)

    debug_canvas = np.hstack([image, filter_panel, result_panel])

    cv2.imwrite(debug_path, debug_canvas)
    print(f"[DEBUG] debug image saved in: {debug_path}")

def read_from_json(input_dir):
    with open(input_dir) as json_data:
        return json.load(json_data)["image_path"]
 
def write_to_json(output_dir, contents_list):
    with open(outputDir, 'w') as f:
        json.dump(contentsList, f) # FIXME: Convert list into dict?
 
def read_image(images, image_index):
    full_path = images[image_index]
 
    img_pool_table = cv2.imread(full_path)
 
    image_name = os.path.basename(full_path)
 
    return img_pool_table, image_name

def order_points_clockwise(pts):
    pts = np.asarray(pts, dtype=np.float32)

    # Handle OpenCV contour format (N, 1, 2) -> (N, 2)
    if pts.ndim == 3 and pts.shape[1] == 1 and pts.shape[2] == 2:
        pts = pts.reshape(-1, 2)

    # Compute centroid of all points
    c = np.mean(pts, axis=0)

    # Compute angle of each point relative to the centroid
    angles = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])

    # Sort points by angle (counterclockwise order)
    pts = pts[np.argsort(angles)]

    # Rotate order so that sequence starts at top-left point
    # (top-left is approximated as the point with smallest x + y)
    s = pts.sum(axis=1)
    start = np.argmin(s)
    pts = np.roll(pts, -start, axis=0)

    return pts.astype(np.float32)

def warp_table(image, image_name, output_dir, debug_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Convert image to HSV and isolate blue table cloth
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 60, 60], dtype=np.uint8)
    upper_blue = np.array([144, 255, 255], dtype=np.uint8)
    blue_mask = cv2.inRange(hsv_img, lower_blue, upper_blue)

    # Find largest blue contour (assumed table region)
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"Blue area not found {image_name}")
        return None

    blue_cnt = max(contours, key=cv2.contourArea)
    blue_only = np.zeros_like(blue_mask)
    cv2.drawContours(blue_only, [blue_cnt], -1, 255, thickness=cv2.FILLED)

    # Build convex hull around detected table and estimate a 4-corner polygon
    hull = cv2.convexHull(blue_cnt)
    perimeter = cv2.arcLength(hull, True)

    quad = None
    best = None
    best_diff = 10**9

    for eps_factor in np.linspace(0.005, 0.10, 120):
        approx = cv2.approxPolyDP(hull, eps_factor * perimeter, True)
        diff = abs(len(approx) - 4)

        if diff < best_diff:
            best = approx
            best_diff = diff

        if len(approx) == 4:
            quad = approx
            break

    # Fallback if no exact 4-point polygon is found
    if quad is None:
        if best is not None and len(best) == 4:
            quad = best
        else:
            rect = cv2.minAreaRect(hull)
            quad = cv2.boxPoints(rect).astype(np.int32).reshape(-1, 1, 2)

    # Create final table mask from detected quadrilateral
    table_mask = np.zeros_like(blue_only)
    cv2.drawContours(table_mask, [quad], -1, 255, thickness=cv2.FILLED)

    # Build result panel (masked table + detected contour)
    result_img = cv2.bitwise_and(image, image, mask=table_mask)
    cv2.drawContours(result_img, [quad], -1, (0, 255, 255), 3)

    # Save debug visualization of the table mask
    visualize_table_mask_debug(
        image=image,
        image_name=image_name,
        output_dir=debug_dir,
        filter_mask=table_mask,
        result_image=result_img,
    )

    # Order corner points consistently (tl, tr, br, bl) top-left, top-right...
    src = order_points_clockwise(quad)
    (tl, tr, br, bl) = src

    # Compute output rectangle size
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(1, int(max(widthA, widthB)))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(1, int(max(heightA, heightB)))

    # Define destination points for bird's-eye view transform
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype=np.float32)

    # Compute perspective transform and warp image
    H = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, H, (maxWidth, maxHeight))

    # Save warped result
    base_name = os.path.splitext(image_name)[0]
    warped_path = os.path.join(output_dir, f"{base_name}_warped_table.jpg")
    cv2.imwrite(warped_path, warped)
    print(f"[INFO] warped table saved in: {warped_path}")

def detect_balls(img):
    return # TODO

def main():
    args = parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    input_json_path = os.path.join(project_root, args.input_json)
    images = read_from_json(input_json_path)

    max_images = min(len(images), args.max_images if args.max_images > 0 else len(images))
    selected_images = images[:max_images]

    output_dir = os.path.join(project_root, args.output_dir)
    debug_dir = os.path.join(project_root, args.debug_dir)

    print(f"[INFO] Processing {max_images} image(s)")
    for image_rel_path in selected_images:
        table_image = cv2.imread(os.path.join(project_root, image_rel_path))
        img_name = os.path.basename(image_rel_path)

        if table_image is None:
            print(f"[WARN] Could not read image: {image_rel_path}")
            continue

        warp_table(table_image, img_name, output_dir=output_dir, debug_dir=debug_dir)


if __name__ == "__main__":
    main()