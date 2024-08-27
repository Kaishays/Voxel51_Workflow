# Import the necessary module
import fiftyone as fo
import fiftyone.types as fot

# Define the paths to your dataset
image_dir = "C:/Git/ml/DataManagement/datasets/06_V3 Bootleg/new val/labels_val_images_640_02"
annotations_path = "C:/Git/ml/DataManagement/datasets/06_V3 Bootleg/new val/labels_val_640_02.json"
#image_dir = "C:/Git/ml/DataManagement/datasets/06_V3 Bootleg/new train/labels_Train_images_640_02"
#annotations_path = "C:/Git/ml/DataManagement/datasets/06_V3 Bootleg/new train/labels_Train_640_02.json"
#image_dir = "C:/Git/ml/DataManagement/datasets/COCO/train2017"
#annotations_path = "C:/Git/ml/DataManagement/datasets/COCO/annotations/instances_train2017.json"
# Create a FiftyOne dataset from the COCO dataset

dataset = fo.Dataset.from_dir(
    dataset_type=fot.COCODetectionDataset,
    data_path=image_dir,
    labels_path=annotations_path,
    name="06_V3Val_NoBuildingClass"
    #name="my_coco_daa00qqtaset" THIS IS THE ACUTAL COCO TRAIN DATASET 
)

#dataset = fo.load_dataset("06_V3Train_NoBuildingClass")
def filter_buildings(sample):
    # Filter 'car' detections only
    if sample.detections:
        sample.detections.detections = [det for det in sample.detections.detections if det.label == "Car"]
    
    # Filter 'car' segmentations only
    if sample.segmentations:
        sample.segmentations.detections = [seg for seg in sample.segmentations.detections if seg.label == "Car"]
    
    # Save changes to the sample
    sample.save()

# Apply the function to each sample in the dataset
for sample in dataset:
    filter_buildings(sample)

if __name__ == "__main__":
        session = fo.launch_app(dataset)
        session.wait()