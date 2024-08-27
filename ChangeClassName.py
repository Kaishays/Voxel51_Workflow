import fiftyone as fo
import fiftyone.core.labels as  fcl
# Load the dataset
dataset = fo.load_dataset("06_V7")

# Define old and new class names
old_class_name = "person"
new_class_name = "car"
label_field = "RT-DETR_Bootleg"

view = {
    dataset
    .filter_labels(label_field, fo.ViewField("label") == old_class_name
)
}
print(len(view))
# Iterate through samples and rename class
count = 0
for sample in view:
    count += 1 
    print(count)
    detections = sample["RT-DETR_Bootleg"].detections
    for detection in detections:
        if detection.label == old_class_name:
            detection.label = new_class_name
    sample.save()

# Save changes to dataset
dataset.save()