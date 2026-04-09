import numpy as np
import cv2
import os
import json

from debug import visualize_table_mask_debug, save_debug_image

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
""" 
def remove_shadows_colored(image, blur_kernel=51):
    # Use LAB luminance normalization to reduce uneven illumination while preserving colors.
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Estimate smooth background illumination from luminance.
    bg = cv2.medianBlur(l_channel, blur_kernel)

    # Normalize luminance against the estimated background.
    l_norm = cv2.divide(l_channel, bg, scale=255)

    normalized = cv2.merge((l_norm, a_channel, b_channel))
    return cv2.cvtColor(normalized, cv2.COLOR_LAB2BGR) """

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

def create_mask(image, image_name, output_dir, debugOutput = True):
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

    if (debugOutput):
        # Save debug visualization of the table mask
        visualize_table_mask_debug(
            image=image,
            image_name=image_name,
            output_dir="output_debug",
            table_mask=table_mask,
        )

    return (table_mask, quad)

def warp_table(image, image_name, output_dir, debugOutput):
    # Create the table mask
    table_mask, quad = create_mask(image, image_name, output_dir, debugOutput)

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
    return warped, H, maxWidth, maxHeight


#
# TODO: remove, already defined in another branch
#
def crop_ball_roi(image, x, y, r, pad=4):
    h, w = image.shape[:2]
    r = int(r) + pad
    x, y = int(x), int(y)

    x1 = max(0, x - r)
    y1 = max(0, y - r)
    x2 = min(w, x + r)
    y2 = min(h, y + r)

    roi = image[y1:y2, x1:x2]
    return roi, (x1, y1, x2, y2)

def detect_balls(image, image_name, output_dir, debugOutput):

    # Milimiters
    POOL_TABLE_LENGTH = 254*10
    POOL_BALL_DIAMETER = 58

    # Diameter of a ball is now aproximatly POOL_TABLE_BALL_RATIO of the width of the table
    POOL_TABLE_BALL_RATIO = POOL_BALL_DIAMETER/POOL_TABLE_LENGTH

    # Warp the table and get the transformation matrix
    warped, H, width, height = warp_table(image, image_name, output_dir, debugOutput)

    # Pool tables have width > height so we can identify the size by checking which is the biggest number:
    if (height > width):
        width = height

    table_mask, quad = create_mask(image, image_name, output_dir, debugOutput)
    ima = cv2.bitwise_and(image, image, mask=table_mask)

    # Before applying the HoughCircles we need to convert the image to grayscale
    # otherwise it will fail. We can also blur the image to remove some of the noise that may interfere.
    gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 7)

    # Detect circles using Hough Transform
    ballRadius = (width*POOL_TABLE_BALL_RATIO)/2.0
    offset = 12

    # Slightly less than a full diameter (2 * radius)
    minDist = ballRadius * 1.8
    param1 = 40
    param2 = 13
    minRadius = int(ballRadius)
    maxRadius = int(ballRadius + offset)

    print(f"[INFO] minDist={minDist}, param1={param1}, param2={param2}, minRadius={minRadius}, maxRadius={maxRadius}")

    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1,  minDist=minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

    # Draw the detected balls on the original angled image
    output_image = image.copy()
    
    # We need the inverse matrix to project coordinates back to original image
    #H_inv = np.linalg.inv(H)
    
    resultingCircles = []

    if circles is not None:
        circles = np.uint16(np.around(circles))
        hsv_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)
        near_black_value_threshold = 62
        for circle_idx, i in enumerate(circles[0, :], start=1):
            x, y, r = i[0], i[1], i[2]

            # Crop the image to just the detected circle.
            (hsv_image_cropped, _) = crop_ball_roi(hsv_image, x, y, 10, pad=0)
            hue_values = hsv_image_cropped[:, :, 0]

            # Find the dominant hue
            hist, bins = np.histogram(hue_values, bins=180, range=(0, 180))
            dominant_hue = np.argmax(hist)
            value_values = hsv_image_cropped[:, :, 2]
            mean_value = float(np.mean(value_values))
            is_near_black = mean_value <= near_black_value_threshold

            if debugOutput:
                base_name = os.path.splitext(image_name)[0]
                hue_debug_path = os.path.join(output_dir, f"{base_name}_hue_image_cropped_{circle_idx:02d}.png")
                cv2.imwrite(hue_debug_path, cv2.cvtColor(hsv_image_cropped, cv2.COLOR_HSV2BGR))
                print(f"[DEBUG] hue_image_cropped saved in: {hue_debug_path} and dominant hue is {dominant_hue}, mean V is {mean_value:.1f}, near_black={is_near_black}")

            if (dominant_hue > 90 and dominant_hue < 103) and (not is_near_black):
                print(f"Color is {dominant_hue}")
                continue

            

            resultingCircles.append([x,y,r])
            

            # Draw the outer circle
            cv2.circle(output_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Draw the center of the circle
            cv2.circle(output_image, (i[0], i[1]), 2, (0, 0, 255), 3)

    # Save output for debugging
    save_debug_image(output_image, image_name, "debug")
    
    return resultingCircles

images = read_from_json(json_input)

#table_image, img_name = read_image(images, 5)
#detect_balls(table_image, img_name, output_dir="output", debugOutput=True)

for i in range(len(images)):
    table_image, img_name = read_image(images, i)

    #warp_table(table_image, img_name, output_dir="output")
    detect_balls(table_image, img_name, output_dir="output", debugOutput=False)