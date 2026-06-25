import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

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

def debug_dominant_color(ball_roi, hsv_img, mask, dominant_color):
    # Display original ROI, mask, hue histogram, and a swatch of the detected dominant color.
    masked_hue = hsv_img[:, :, 0][mask.astype(bool)]
    plt.figure(figsize=(18, 4))
    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(ball_roi, cv2.COLOR_BGR2RGB))
    plt.title("Original Ball ROI")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(mask, cmap='gray')
    plt.title("Mask Used for Color Detection")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.hist(masked_hue, bins=180, range=(0, 180), color='blue', alpha=0.7)
    h, s, v = dominant_color[0][0]
    plt.axvline(h, color='red', linestyle='--', label=f'Dominant Hue: {h}')
    plt.title("Hue Histogram of Masked Region")
    plt.xlabel("Hue Value")
    plt.ylabel("Pixel Count")
    plt.legend()

    plt.subplot(1, 4, 4)
    if dominant_color is None:
        swatch_rgb = np.full((120, 120, 3), 128, dtype=np.uint8)
        swatch_title = "Dominant Color: N/A"
    else:
        h, s, v = dominant_color[0][0]
        swatch_rgb = cv2.cvtColor(dominant_color, cv2.COLOR_HSV2RGB)[0][0]
        swatch_rgb = np.full((120, 120, 3), swatch_rgb, dtype=np.uint8)
        swatch_title = f"Dominant Color (H={h}, S={s}, V={v})"
    plt.imshow(swatch_rgb)
    plt.title(swatch_title)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

""" Visualize the detected balls by drawing bounding boxes and placing the detected number below each ball ROI """
def show_detected_balls_on_image(image, balls):
    display_img = image.copy()

    for ball in balls:
        (x1, y1, x2, y2) = ball.bbox
        number = ball.number

        # Draw ROI bounding box
        cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Place the detected number slightly below the ROI
        label = "?" if number is None else str(number)
        text_x = x1
        text_y = min(display_img.shape[0] - 5, y2 + 20)
        cv2.putText(display_img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    f, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
    ax.set_title("Detected Balls (Number Under ROI)")
    ax.axis("off")
    plt.show()