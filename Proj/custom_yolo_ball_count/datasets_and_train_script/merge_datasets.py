import os
import json
import shutil
import math

"""
Merges all pool ball datasets into a single unified JSON.
Output format example:

[
  {
    "id": 0,
    "file_name": "some_image.jpg",
    "image_path": "datasets/merged/images/some_image.jpg",
    "width": 640,
    "height": 640,
    "source": "Pool Ball Detection.v5i.coco/train",
    "balls": [
      {"x": 378, "y": 225, "width": 17, "height": 16},
      {"x": 480, "y": 214, "width": 16, "height": 15}
    ]
  },
  ...
]

Bounding box values (x, y, width, height) are in pixel coordinates.
"""

INPUT = "datasets/original"
OUTPUT = "datasets/merged"
OUTPUT_IMAGES = f"{OUTPUT}/images"
OUTPUT_JSON = f"{OUTPUT}/_annotations.json"

# This contains the datasets and the splits that are available. The set defined what classes should be ignored...
DATASETS = [
    ("8-Ball Pool.v3i.coco", ["train"], {"Dot"}),
    ("Pool Ball Detection.v5i.coco", ["train", "test", "valid"], set()),
    ("Pool Balls Detection.v13-v13.coco", ["train", "test", "valid"], set()),
    ("Pool Billiard.v1i.coco", ["train", "test", "valid"], set()),
]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


os.makedirs(OUTPUT_IMAGES, exist_ok=True)

merged = []
seen_filenames = set()
skipped_no_image = 0
skipped_duplicate = 0
skipped_no_balls = 0
global_id = 0

for dataset_name, splits, *rest in DATASETS:
    exclude_categories = rest[0] if rest else set()

    for split in splits:
        coco_path = f"{INPUT}/{dataset_name}/{split}/_annotations.coco.json"
        images_dir = f"{INPUT}/{dataset_name}/{split}"

        if not os.path.exists(coco_path):
            print(f"  skip (missing): {coco_path}")
            continue

        coco = load_json(coco_path)

        # Build set of excluded category ids from their names
        excluded_ids = {
            c["id"] for c in coco.get("categories", [])
            if c["name"] in exclude_categories
        }

        # Groups annotations by image_id and it also drops any excluded categories.
        balls_by_image_id = {}
        for ann in coco["annotations"]:
            if ann["category_id"] not in excluded_ids:
                balls_by_image_id.setdefault(ann["image_id"], []).append(ann)

        added = 0
        for img in coco["images"]:
            file_name = img["file_name"]

            if file_name in seen_filenames:
                skipped_duplicate += 1
                continue
            seen_filenames.add(file_name)

            balls = [
                {"x": ann["bbox"][0], "y": ann["bbox"][1], "width": round(ann["bbox"][2]), "height": round(ann["bbox"][3])}
                for ann in balls_by_image_id.get(img["id"], [])
            ]

            if not balls:
                skipped_no_balls += 1
                continue

            src = os.path.join(images_dir, file_name)
            if not os.path.exists(src):
                skipped_no_image += 1
                print(f"  image file missing: {src}")
                continue

            dst = os.path.join(OUTPUT_IMAGES, file_name)
            shutil.copy2(src, dst)

            merged.append({
                "id": global_id,
                "file_name": file_name,
                "image_path": dst,
                "width": img.get("width", 0),
                "height": img.get("height", 0),
                "source": f"{dataset_name}/{split}",
                "balls": balls,
            })
            global_id += 1
            added += 1

        print(f"  {dataset_name}/{split}: {added} images added")

write_json(OUTPUT_JSON, merged)

total_balls = sum(len(r["balls"]) for r in merged)
print(f"\nDone.")
print(f"  Images:          {len(merged)}")
print(f"  Total balls:     {total_balls}")
print(f"  Avg balls/image: {total_balls / len(merged):.1f}")
print(f"  Skipped (dup):   {skipped_duplicate}")
print(f"  Skipped (no img):{skipped_no_image}")
print(f"  Skipped (no ann):{skipped_no_balls}")
print(f"  Output:          {OUTPUT_JSON}")


new_merged_annotations = load_json(OUTPUT_JSON)

print(new_merged_annotations[0])