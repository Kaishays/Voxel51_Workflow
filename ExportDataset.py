import fiftyone as fo
import fiftyone.utils.data.exporters as foude


export_dir = "C:/Git/ml/DataManagement/datasets/06/06_V7/Test(.2)"

dataset_name = "06_V7"
label_field = "GTV1_BL1_101COCO" 

dataset = fo.load_dataset(dataset_name)
saved_view_name = "GTV1_BL1_101COCO Final"
#"trainV2","fiV2","valV2"
view = (
    dataset
    .load_saved_view(saved_view_name)
    .match_tags(["testV2"], bool=True, all = False) 
)
print(len(view))
foude.export_samples(
    samples=view,
    export_dir=export_dir,
    dataset_type=fo.types.COCODetectionDataset,
    label_field=label_field,
)

'''''
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
'''