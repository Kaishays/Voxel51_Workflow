import fiftyone as fo
import fiftyone.utils.data.exporters as foude
from fiftyone import ViewField as F
import random

export_dir = "C:/Git/ml/DataManagement/datasets/06_V5_Temp"
#export_dir = "C:/Git/ml/DataManagement/datasets/06_V3 Bootleg/Train_NoBuildingClass"

dataset_name = "06_V3"
label_field = "TempExportLabels"

dataset = fo.load_dataset(dataset_name)

view = (
    dataset
    .match_tags(["Car", "Building", "Person", "IR", "Low bitrate"], bool=False, all = False)
    #.match(F(label_field).exists())
)
selection_probability = .6442 / 1.2083
total_samples = len(view)
print(total_samples)
num_samples_to_export = int(total_samples * selection_probability)


random_view = dataset.take(num_samples_to_export)




count = 0
bbox = [0, 0, 10, 10]
with fo.ProgressBar() as pb:
    for sample in random_view:
        count += 1
        print(count)
        detections = []
        detection = fo.Detection(label="temp", bounding_box=bbox, confidence=0.5)
        detections.append(detection)
        sample["TempExportLabels"] = fo.Detections(detections=detections)
        sample.save()


foude.export_samples(
    samples=random_view,
    export_dir=export_dir,
    dataset_type=fo.types.COCODetectionDataset,
    label_field=label_field,
)
