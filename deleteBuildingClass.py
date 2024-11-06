import fiftyone as fo
import fiftyone.types as fot

image_dir = "C:/Git/ml/DataManagement/datasets/06_V3 Bootleg/new val/labels_val_images_640_02"
annotations_path = "C:/Git/ml/DataManagement/datasets/06_V3 Bootleg/new val/labels_val_640_02.json"

dataset = fo.Dataset.from_dir(
    dataset_type=fot.COCODetectionDataset,
    data_path=image_dir,
    labels_path=annotations_path,
    name="06_V3Val_NoBuildingClass"
)

def filter_buildings(sample):
    if sample.detections:
        sample.detections.detections = [det for det in sample.detections.detections if det.label == "Car"]
    
    if sample.segmentations:
        sample.segmentations.detections = [seg for seg in sample.segmentations.detections if seg.label == "Car"]
    
    sample.save()

for sample in dataset:
    filter_buildings(sample)

if __name__ == "__main__":
        session = fo.launch_app(dataset)
        session.wait()