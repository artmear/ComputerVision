import numpy as np
import cv2
import os
import json

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

def get_correct_orientation(warped_square_img):
    h, w = warped_square_img.shape[:2]
    gray = cv2.cvtColor(warped_square_img, cv2.COLOR_BGR2GRAY)

    margin = max(5, int(min(h, w) * 0.01))
    half_roi = max(7, int(min(h, w) * 0.015))
    depth = max(15, int(min(h, w) * 0.03))

    x0 = max(0, w // 2 - half_roi)
    x1 = min(w, w // 2 + half_roi)
    y0 = max(0, h // 2 - half_roi)
    y1 = min(h, h // 2 + half_roi)

    boxes = {
        "top": (x0, margin, x1, margin + depth),
        "bottom": (x0, h - margin - depth, x1, h - margin),
        "left": (margin, y0, margin + depth, y1),
        "right": (w - margin - depth, y0, w - margin, y1),
    }

    mid_top = gray[boxes["top"][1]:boxes["top"][3], boxes["top"][0]:boxes["top"][2]]
    mid_bottom = gray[boxes["bottom"][1]:boxes["bottom"][3], boxes["bottom"][0]:boxes["bottom"][2]]
    mid_left = gray[boxes["left"][1]:boxes["left"][3], boxes["left"][0]:boxes["left"][2]]
    mid_right = gray[boxes["right"][1]:boxes["right"][3], boxes["right"][0]:boxes["right"][2]]

    score_horizontal = float(np.mean(mid_top) + np.mean(mid_bottom))
    score_vertical = float(np.mean(mid_left) + np.mean(mid_right))

    orientation = "HORIZONTAL" if score_horizontal < score_vertical else "VERTICAL"
    return orientation, score_horizontal, score_vertical, boxes

def _to_bgr(img):
    if img is None:
        return None
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def _make_debug_tile(img, label, tile_w=560, tile_h=360):
    bgr = _to_bgr(img)
    if bgr is None:
        bgr = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    tile = cv2.resize(bgr, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
    cv2.rectangle(tile, (0, 0), (tile_w, 42), (0, 0, 0), thickness=cv2.FILLED)
    cv2.putText(tile, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    return tile

def _compose_debug_grid(tiles, columns=2):
    if not tiles:
        return np.zeros((360, 560, 3), dtype=np.uint8)

    tile_h, tile_w = tiles[0].shape[:2]
    rows = int(np.ceil(len(tiles) / columns))
    total_slots = rows * columns

    if len(tiles) < total_slots:
        blank = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
        tiles = tiles + [blank] * (total_slots - len(tiles))

    row_imgs = []
    for row_idx in range(rows):
        start = row_idx * columns
        end = start + columns
        row_imgs.append(cv2.hconcat(tiles[start:end]))

    return cv2.vconcat(row_imgs)

def save_debug_pipeline(
    image,
    image_name,
    output_dir,
    blue_mask,
    blue_only,
    hull,
    quad,
    table_mask,
    warped_probe,
    warped_final,
    orientation,
    score_horizontal,
    score_vertical,
    sample_boxes,
):
    os.makedirs(output_dir, exist_ok=True)

    contour_stage = image.copy()
    cv2.drawContours(contour_stage, [hull], -1, (0, 255, 0), 2)
    cv2.drawContours(contour_stage, [quad], -1, (0, 0, 255), 3)

    probe_stage = warped_probe.copy()
    for name, (x0, y0, x1, y1) in sample_boxes.items():
        color = (0, 255, 0) if name in ("top", "bottom") else (0, 165, 255)
        cv2.rectangle(probe_stage, (x0, y0), (x1, y1), color, 2)

    probe_text = f"ori={orientation} H={score_horizontal:.1f} V={score_vertical:.1f}"
    cv2.rectangle(probe_stage, (0, 0), (probe_stage.shape[1], 28), (0, 0, 0), thickness=cv2.FILLED)
    cv2.putText(probe_stage, probe_text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    tiles = [
        _make_debug_tile(image, "01 Original"),
        _make_debug_tile(blue_mask, "02 Blue Mask"),
        _make_debug_tile(blue_only, "03 Largest Blue"),
        _make_debug_tile(contour_stage, "04 Hull + Quad"),
        _make_debug_tile(table_mask, "05 Table Mask"),
        _make_debug_tile(probe_stage, "06 Warp Probe"),
        _make_debug_tile(warped_final, "07 Final Warp"),
    ]

    debug_panel = _compose_debug_grid(tiles, columns=2)
    base_name = os.path.splitext(image_name)[0]
    debug_path = os.path.join(output_dir, f"{base_name}_debug_pipeline.jpg")
    cv2.imwrite(debug_path, debug_panel)
    print(f"[DEBUG] pipeline saved in: {debug_path}")

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

    # Order corner points consistently (tl, tr, br, bl) top-left, top-right...
    src = order_points_clockwise(quad)
    (tl, tr, br, bl) = src

    # Provisional square warp to infer orientation from middle-pocket darkness
    probe_side = 500
    dst_probe = np.array([
        [0, 0],
        [probe_side - 1, 0],
        [probe_side - 1, probe_side - 1],
        [0, probe_side - 1]
    ], dtype=np.float32)
    H_probe = cv2.getPerspectiveTransform(src, dst_probe)
    warped_probe = cv2.warpPerspective(image, H_probe, (probe_side, probe_side))
    orientation, score_horizontal, score_vertical, sample_boxes = get_correct_orientation(warped_probe)

    # Compute output size as 1:1 (square)
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    side = max(1, int(max(widthA, widthB, heightA, heightB)))

    # Define destination points for bird's-eye view transform
    dst = np.array([
        [0, 0],
        [side - 1, 0],
        [side - 1, side - 1],
        [0, side - 1]
    ], dtype=np.float32)

    # Compute perspective transform and warp image
    H = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, H, (side, side))

    # If middle pockets were detected at left/right, rotate to horizontal layout
    if orientation == "VERTICAL":
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    save_debug_pipeline(
        image=image,
        image_name=image_name,
        output_dir="output_debug",
        blue_mask=blue_mask,
        blue_only=blue_only,
        hull=hull,
        quad=quad,
        table_mask=table_mask,
        warped_probe=warped_probe,
        warped_final=warped,
        orientation=orientation,
        score_horizontal=score_horizontal,
        score_vertical=score_vertical,
        sample_boxes=sample_boxes,
    )

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