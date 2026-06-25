"""
Convert a COCO-format annotation JSON to the balls detection output format.
 
COCO bbox format: [x, y, width, height] in pixels
Output format:    normalized [xmin, xmax, ymin, ymax] as fractions of image size
 
Category → ball number mapping (billiards convention):
  Cue    → 0
  Solid  → 1-7  (assigned in annotation order per image)
  Black  → 8
  Striped → 9-15 (assigned in annotation order per image)
  Dot    → skipped (pocket/table markers, not balls)
 
Usage:
  python convert_coco_to_balls.py <input_coco.json> <output.json>
  python convert_coco_to_balls.py annotations_coco.json output.json
"""
 
import json
import argparse
from collections import defaultdict
 
 
# Ball number assignment per category.
# Cue and Black have fixed numbers; Solid/Striped get sequential numbers
# within their group, assigned in annotation order per image.
FIXED_NUMBER = {
    "Cue": 0,
    "Black": 8,
}
SEQUENTIAL_RANGES = {
    "Solid": list(range(1, 8)),    # 1-7
    "Striped": list(range(9, 16)), # 9-15
}
SKIP_CATEGORIES = {"Dot", "balls"}  # table markers / root category
 
 
def convert(input_path: str, output_path: str) -> None:
    with open(input_path, "r") as f:
        coco = json.load(f)
 
    # Build lookup maps
    image_map = {img["id"]: img for img in coco["images"]}
    category_map = {cat["id"]: cat["name"] for cat in coco["categories"]}
 
    # Group annotations by image
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)
 
    results = []
 
    for image_id, annotations in anns_by_image.items():
        img = image_map[image_id]
        width = img["width"]
        height = img["height"]
        image_path = img["file_name"]
 
        balls = []
        # Track how many Solid/Striped balls have been assigned per image
        sequential_counters = {cat: 0 for cat in SEQUENTIAL_RANGES}
 
        for ann in annotations:
            cat_name = category_map.get(ann["category_id"], "")
 
            if cat_name in SKIP_CATEGORIES:
                continue
 
            # Determine ball number
            if cat_name in FIXED_NUMBER:
                ball_number = FIXED_NUMBER[cat_name]
            elif cat_name in SEQUENTIAL_RANGES:
                idx = sequential_counters[cat_name]
                available = SEQUENTIAL_RANGES[cat_name]
                if idx >= len(available):
                    # More balls than expected — append beyond range
                    ball_number = available[-1] + (idx - len(available) + 1)
                else:
                    ball_number = available[idx]
                sequential_counters[cat_name] += 1
            else:
                # Unknown category: skip
                continue
 
            # COCO bbox: [x, y, w, h] — values may be stored as strings
            x, y, w, h = (float(v) for v in ann["bbox"])
 
            xmin = x / width
            xmax = (x + w) / width
            ymin = y / height
            ymax = (y + h) / height
 
            balls.append({
                "number": ball_number,
                "xmin": xmin,
                "xmax": xmax,
                "ymin": ymin,
                "ymax": ymax,
            })
 
        results.append({
            "image_path": image_path,
            "num_balls": len(balls),
            "balls": balls,
        })
 
    # Sort output by image_path for deterministic ordering
    results.sort(key=lambda r: r["image_path"])
 
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
 
    print(f"Converted {len(results)} images → {output_path}")
 
 
def main():
    parser = argparse.ArgumentParser(description="Convert COCO JSON to balls output format")
    parser.add_argument("input", help="Path to COCO annotations JSON")
    parser.add_argument("output", help="Path for output JSON")
    args = parser.parse_args()
    convert(args.input, args.output)
 
 
if __name__ == "__main__":
    main()