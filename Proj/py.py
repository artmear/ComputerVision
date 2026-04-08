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
    parser.add_argument("--hsv-lower", type=int, nargs=3, default=[90, 60, 60])
    parser.add_argument("--hsv-upper", type=int, nargs=3, default=[144, 255, 255])
    parser.add_argument("--approx-eps-min", type=float, default=0.005)
    parser.add_argument("--approx-eps-max", type=float, default=0.10)
    parser.add_argument("--approx-eps-steps", type=int, default=120)
    parser.add_argument("--result-contour-thickness", type=int, default=3)
    parser.add_argument("--interactive", action="store_true")
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

def detect_table_geometry(image, args):
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array(args.hsv_lower, dtype=np.uint8)
    upper_blue = np.array(args.hsv_upper, dtype=np.uint8)
    blue_mask = cv2.inRange(hsv_img, lower_blue, upper_blue)

    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return blue_mask, None

    blue_cnt = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(blue_cnt)
    perimeter = cv2.arcLength(hull, True)

    quad = None
    best = None
    best_diff = 10**9

    for eps_factor in np.linspace(args.approx_eps_min, args.approx_eps_max, args.approx_eps_steps):
        approx = cv2.approxPolyDP(hull, eps_factor * perimeter, True)
        diff = abs(len(approx) - 4)

        if diff < best_diff:
            best = approx
            best_diff = diff

        if len(approx) == 4:
            quad = approx
            break

    if quad is None:
        if best is not None and len(best) == 4:
            quad = best
        else:
            rect = cv2.minAreaRect(hull)
            quad = cv2.boxPoints(rect).astype(np.int32).reshape(-1, 1, 2)

    return blue_mask, quad

def build_table_outputs(image, args):
    blue_mask, quad = detect_table_geometry(image, args)
    table_mask = np.zeros_like(blue_mask)

    if quad is None:
        result_img = image.copy()
        warped = None
        return table_mask, result_img, warped, None

    cv2.drawContours(table_mask, [quad], -1, 255, thickness=cv2.FILLED)

    result_img = cv2.bitwise_and(image, image, mask=table_mask)
    cv2.drawContours(result_img, [quad], -1, (0, 255, 255), args.result_contour_thickness)

    src = order_points_clockwise(quad)
    (tl, tr, br, bl) = src

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(1, int(max(widthA, widthB)))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(1, int(max(heightA, heightB)))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, H, (maxWidth, maxHeight))

    return table_mask, result_img, warped, quad

def warp_table(image, image_name, output_dir, debug_dir, args):
    os.makedirs(output_dir, exist_ok=True)
    table_mask, result_img, warped, quad = build_table_outputs(image, args)

    if quad is None or warped is None:
        print(f"Blue area not found {image_name}")
        return None

    # Save debug visualization of the table mask
    visualize_table_mask_debug(
        image=image,
        image_name=image_name,
        output_dir=debug_dir,
        filter_mask=table_mask,
        result_image=result_img,
    )

    # Save warped result
    base_name = os.path.splitext(image_name)[0]
    warped_path = os.path.join(output_dir, f"{base_name}_warped_table.jpg")
    cv2.imwrite(warped_path, warped)
    print(f"[INFO] warped table saved in: {warped_path}")

def interactive_adjust(image, image_name, output_dir, debug_dir, args):
    control_window = "Controls"
    preview_window = "Preview"
    info_window = "Controls Info"

    cv2.namedWindow(control_window, cv2.WINDOW_NORMAL)
    cv2.namedWindow(preview_window, cv2.WINDOW_NORMAL)
    cv2.namedWindow(info_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(control_window, 520, 420)
    cv2.resizeWindow(info_window, 520, 360)

    cv2.createTrackbar("H low", control_window, int(args.hsv_lower[0]), 179, lambda v: None)
    cv2.createTrackbar("S low", control_window, int(args.hsv_lower[1]), 255, lambda v: None)
    cv2.createTrackbar("V low", control_window, int(args.hsv_lower[2]), 255, lambda v: None)
    cv2.createTrackbar("H high", control_window, int(args.hsv_upper[0]), 179, lambda v: None)
    cv2.createTrackbar("S high", control_window, int(args.hsv_upper[1]), 255, lambda v: None)
    cv2.createTrackbar("V high", control_window, int(args.hsv_upper[2]), 255, lambda v: None)
    cv2.createTrackbar("eps min x1000", control_window, int(args.approx_eps_min * 1000), 300, lambda v: None)
    cv2.createTrackbar("eps max x1000", control_window, int(args.approx_eps_max * 1000), 500, lambda v: None)
    cv2.createTrackbar("eps steps", control_window, int(args.approx_eps_steps), 500, lambda v: None)
    cv2.createTrackbar("contour thick", control_window, int(args.result_contour_thickness), 20, lambda v: None)

    print("[INTERACTIVE] Keys: s=save, n=next image, q/esc=quit")

    while True:
        h_low = cv2.getTrackbarPos("H low", control_window)
        s_low = cv2.getTrackbarPos("S low", control_window)
        v_low = cv2.getTrackbarPos("V low", control_window)
        h_high = cv2.getTrackbarPos("H high", control_window)
        s_high = cv2.getTrackbarPos("S high", control_window)
        v_high = cv2.getTrackbarPos("V high", control_window)

        if h_high < h_low:
            h_high = h_low
            cv2.setTrackbarPos("H high", control_window, h_high)
        if s_high < s_low:
            s_high = s_low
            cv2.setTrackbarPos("S high", control_window, s_high)
        if v_high < v_low:
            v_high = v_low
            cv2.setTrackbarPos("V high", control_window, v_high)

        eps_min_i = max(1, cv2.getTrackbarPos("eps min x1000", control_window))
        eps_max_i = max(1, cv2.getTrackbarPos("eps max x1000", control_window))
        if eps_max_i < eps_min_i:
            eps_max_i = eps_min_i
            cv2.setTrackbarPos("eps max x1000", control_window, eps_max_i)

        eps_steps = max(2, cv2.getTrackbarPos("eps steps", control_window))
        if eps_steps != cv2.getTrackbarPos("eps steps", control_window):
            cv2.setTrackbarPos("eps steps", control_window, eps_steps)

        contour_thickness = max(1, cv2.getTrackbarPos("contour thick", control_window))
        if contour_thickness != cv2.getTrackbarPos("contour thick", control_window):
            cv2.setTrackbarPos("contour thick", control_window, contour_thickness)

        args.hsv_lower = [h_low, s_low, v_low]
        args.hsv_upper = [h_high, s_high, v_high]
        args.approx_eps_min = eps_min_i / 1000.0
        args.approx_eps_max = eps_max_i / 1000.0
        args.approx_eps_steps = eps_steps
        args.result_contour_thickness = contour_thickness

        table_mask, result_img, warped, _ = build_table_outputs(image, args)

        mask_panel = cv2.cvtColor(table_mask, cv2.COLOR_GRAY2BGR)
        if warped is None:
            warped_panel = np.zeros_like(image)
        else:
            warped_panel = cv2.resize(warped, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

        top = np.hstack([image, mask_panel])
        bottom = np.hstack([result_img, warped_panel])
        preview = np.vstack([top, bottom])

        cv2.putText(preview, "Original | Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(preview, "Result | Warped", (10, image.shape[0] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        info_y = 65
        cv2.putText(preview, f"HSV low: {args.hsv_lower}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(preview, f"HSV high: {args.hsv_upper}", (10, info_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(preview, f"eps min/max: {args.approx_eps_min:.3f} / {args.approx_eps_max:.3f}", (10, info_y + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(preview, f"eps steps: {args.approx_eps_steps} | contour: {args.result_contour_thickness}", (10, info_y + 84), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.imshow(preview_window, preview)

        info_panel = np.zeros((360, 520, 3), dtype=np.uint8)
        cv2.putText(info_panel, "Interactive Parameters", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
        cv2.putText(info_panel, f"H low: {h_low}      H high: {h_high}", (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(info_panel, f"S low: {s_low}      S high: {s_high}", (12, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(info_panel, f"V low: {v_low}      V high: {v_high}", (12, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(info_panel, f"eps min: {args.approx_eps_min:.3f}", (12, 186), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(info_panel, f"eps max: {args.approx_eps_max:.3f}", (12, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(info_panel, f"eps steps: {args.approx_eps_steps}", (12, 254), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(info_panel, f"contour thickness: {args.result_contour_thickness}", (12, 288), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(info_panel, "Keys: s=save, n=next, q/esc=quit", (12, 334), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (180, 180, 180), 2)
        cv2.imshow(info_window, info_panel)

        key = cv2.waitKey(30) & 0xFF

        if key == ord('s'):
            visualize_table_mask_debug(image, image_name, debug_dir, table_mask, result_img)
            if warped is not None:
                base_name = os.path.splitext(image_name)[0]
                warped_path = os.path.join(output_dir, f"{base_name}_warped_table.jpg")
                cv2.imwrite(warped_path, warped)
                print(f"[INFO] warped table saved in: {warped_path}")
            else:
                print(f"[WARN] Nothing to save, table not detected in {image_name}")
        elif key == ord('n'):
            visualize_table_mask_debug(image, image_name, debug_dir, table_mask, result_img)
            if warped is not None:
                base_name = os.path.splitext(image_name)[0]
                warped_path = os.path.join(output_dir, f"{base_name}_warped_table.jpg")
                cv2.imwrite(warped_path, warped)
                print(f"[INFO] warped table saved in: {warped_path}")
            else:
                print(f"[WARN] Going next without warped output for {image_name}")
            return True
        elif key == ord('q') or key == 27:
            return False

def detect_balls(img):
    return # TODO

def main():
    args = parse_args()

    if args.approx_eps_steps < 2:
        raise ValueError("--approx-eps-steps must be >= 2")
    if args.approx_eps_min <= 0 or args.approx_eps_max <= 0:
        raise ValueError("--approx-eps-min and --approx-eps-max must be > 0")
    if args.approx_eps_min > args.approx_eps_max:
        raise ValueError("--approx-eps-min cannot be greater than --approx-eps-max")
    if args.result_contour_thickness < 1:
        raise ValueError("--result-contour-thickness must be >= 1")

    project_root = os.path.dirname(os.path.abspath(__file__))
    input_json_path = os.path.join(project_root, args.input_json)
    images = read_from_json(input_json_path)

    max_images = min(len(images), args.max_images if args.max_images > 0 else len(images))
    selected_images = images[:max_images]

    output_dir = os.path.join(project_root, args.output_dir)
    debug_dir = os.path.join(project_root, args.debug_dir)

    print(f"[INFO] Processing {max_images} image(s)")
    if args.interactive:
        for image_rel_path in selected_images:
            table_image = cv2.imread(os.path.join(project_root, image_rel_path))
            img_name = os.path.basename(image_rel_path)

            if table_image is None:
                print(f"[WARN] Could not read image: {image_rel_path}")
                continue

            print(f"[INTERACTIVE] Image: {img_name}")
            keep_running = interactive_adjust(table_image, img_name, output_dir=output_dir, debug_dir=debug_dir, args=args)
            if not keep_running:
                break

        cv2.destroyAllWindows()
    else:
        for image_rel_path in selected_images:
            table_image = cv2.imread(os.path.join(project_root, image_rel_path))
            img_name = os.path.basename(image_rel_path)

            if table_image is None:
                print(f"[WARN] Could not read image: {image_rel_path}")
                continue

            warp_table(table_image, img_name, output_dir=output_dir, debug_dir=debug_dir, args=args)


if __name__ == "__main__":
    main()