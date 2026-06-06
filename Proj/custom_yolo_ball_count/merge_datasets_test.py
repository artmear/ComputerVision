"""
What we found was that:

8-Ball Pool.v3i.coco/test:
    The image annotations also considered the small dots around the pool table. Since these had a class of their own they were removed from the merged annotations.

Pool Balls Detection.v13-v13.coco/train:
    This dataset contained values for width and height that were non integers. To fix this we rounded the values considering the fact that it may lead to a sligtly worse model.

    

"""


import json
import cv2

def load_json(path):
    with open(path) as f:
        return json.load(f)

INPUT = "datasets/original"
OUTPUT = "datasets/merged"
OUTPUT_IMAGES = f"{OUTPUT}/images"
OUTPUT_JSON = f"{OUTPUT}/_annotations.json"
new_merged_annotations = load_json(OUTPUT_JSON)

possible_datasets = [
    ("8-Ball Pool.v3i.coco", ["train"]),
    ("Pool Ball Detection.v5i.coco", ["train", "test", "valid"]),
    ("Pool Balls Detection.v13-v13.coco", ["train", "test", "valid"]),
    ("Pool Billiard.v1i.coco", ["train", "test", "valid"]),
]

# This will be used to check if a given dataset was already checked.
# For each possible dataset we are setting it to unvisited.
dataset_visited = {}
for dataset_name, dataset_type in possible_datasets:
    for type in dataset_type:
        source = f"{dataset_name}/{type}"
        dataset_visited[source] = False

#print(dataset_visited)



for annotation in new_merged_annotations:

    dataset_source = annotation["source"]

    if (dataset_visited[dataset_source]):
        continue

    #print(f"Not yet visited - {dataset_source}")
    
    print("The annotation for the image that will be shown is:")
    print(annotation)
    print("\n")
    
    image = cv2.imread(annotation["image_path"])
    
    for ball in annotation["balls"]:
        x_min = ball["x"]
        y_min = ball["y"]
        width = ball["width"]
        height = ball["height"]
    
        start_point = (x_min, y_min)
        end_point = (x_min + width, y_min + height)
        line_color = (255, 255, 0)
        thickness = 2
    
        image = cv2.rectangle(image, start_point, end_point, line_color, thickness)
    
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Set dataset as visited. We only want to check one image of each dataset/datset_type
    dataset_visited[dataset_source] = True