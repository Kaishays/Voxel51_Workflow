import fiftyone as fo

dataset = fo.load_dataset("06_V7")

old_class_name = "person"
new_class_name = "car"
label_field = "RT-DETR_Bootleg"

view = {
    dataset
    .filter_labels(label_field, fo.ViewField("label") == old_class_name
)
}
print(len(view))

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