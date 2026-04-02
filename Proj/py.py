import numpy as np
import cv2
import os
import json

from debug import visualize_table_mask_debug

json_input = "example_json/input.json"

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

def warp_table(image, image_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    h, w = image.shape[:2]

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

    # Save debug visualization of the table mask
    visualize_table_mask_debug(
        image=image,
        image_name=image_name,
        output_dir="output_debug",
        table_mask=table_mask,
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

# def detect_table(image):
#     mask_img = cv2.inRange(hsv_img, lower_blue, upper_blue)

#     # TODO: Remove? This should remove noise but doesn't make any difference
#     #kernel = np.ones((5,5), np.uint8)
#     #mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_CLOSE, kernel)
#     #mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_OPEN, kernel)

#     result_img = cv2.bitwise_and(hsv_img, hsv_img, mask=mask_img)

#     # 2. Convert the colors again
#     result_img = cv2.cvtColor(result_img, cv2.COLOR_HSV2RGB)
#     result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2GRAY)

#     # 3. Apply Hough circles to detect balls
#     result_img = cv2.medianBlur(result_img, 5)
    
#     circles = cv2.HoughCircles(result_img, cv2.HOUGH_GRADIENT, dp=1, minDist=60, param1=50, param2=30, minRadius=15, maxRadius=50)

#     circles = np.uint16(np.around(circles))
#     for i in circles[0,:]:
#         # draw the outer circle
#         cv2.circle(result_img,(i[0],i[1]),i[2],(255,255,0),10)


images = read_from_json(json_input)

# table_image, img_name = read_image(images, 0)

# warp_table(table_image, img_name, output_dir="output")

for i in range(len(images)):
    table_image, img_name = read_image(images, i)

    warp_table(table_image, img_name, output_dir="output")