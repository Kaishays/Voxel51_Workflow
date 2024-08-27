import cv2
import json
import os

# Function to split an image into 640x640 overlapping segments
def split_image(image, image_id, step=640, overlap=320):
    h, w, _ = image.shape
    segments = []
    for y in range(0, h, overlap):
        for x in range(0, w, overlap):
            x_end = min(x + step, w)
            y_end = min(y + step, h)
            if x_end - x == step and y_end - y == step:
                segment = image[y:y_end, x:x_end]
                segments.append((segment, (x, y), image_id))
    return segments

# Function to adjust bounding boxes for the new segments
def adjust_annotations(annotations, segments, original_image_size=(640, 640)):
    new_annotations = []
    for segment, (x_offset, y_offset), image_id in segments:
        for ann in annotations:
            bbox = ann['bbox']
            x, y, w, h = bbox
            # Check if the bounding box is within the current segment
            if x + w > x_offset and x < x_offset + original_image_size[0] and \
               y + h > y_offset and y < y_offset + original_image_size[1]:
                # Adjust the bounding box to fit the new segment
                new_x = max(0, x - x_offset)
                new_y = max(0, y - y_offset)
                new_w = min(w, x_offset + original_image_size[0] - x) - new_x
                new_h = min(h, y_offset + original_image_size[1] - y) - new_y

                new_ann = ann.copy()
                new_ann['bbox'] = [new_x, new_y, new_w, new_h]
                new_ann['image_id'] = f"{image_id}_{x_offset}_{y_offset}"
                new_annotations.append(new_ann)
    return new_annotations

# Load COCO annotations
with open("C:/Git/ml/DataManagement/datasets/06_V3 Bootleg/Final/labels_Final.json", 'r') as f:
    coco_data = json.load(f)

# Output directory
output_dir = "C:/Git/ml/DataManagement/datasets/06_V3 Bootleg/new final"
os.makedirs(output_dir, exist_ok=True)

new_images = []
new_annotations = []
count = 0

# Process each image in the COCO dataset
for image_info in coco_data['images']:
    print(count)
    count += 1
    image_path = os.path.join("C:/Git/ml/DataManagement/datasets/06_V3 Bootleg/Final/images", image_info['file_name'])
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Failed to load image: {image_path}")
        continue
    
    image_id = image_info['id']

    segments = split_image(image, image_id)
    image_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]
    adjusted_annotations = adjust_annotations(image_annotations, segments)

    for i, (segment, (x_offset, y_offset), _) in enumerate(segments):
        segment_file_name = f"{image_id}_{x_offset}_{y_offset}.jpg"
        cv2.imwrite(os.path.join(output_dir, segment_file_name), segment)
        new_images.append({
            'file_name': segment_file_name,
            'height': segment.shape[0],
            'width': segment.shape[1],
            'id': f"{image_id}_{x_offset}_{y_offset}"
        })

    new_annotations.extend(adjusted_annotations)

# Save the new COCO annotations
coco_data['images'] = new_images
coco_data['annotations'] = new_annotations

with open("C:/Git/ml/DataManagement/datasets/06_V3 Bootleg/new final/annotations.json", 'w') as f:
    json.dump(coco_data, f)
