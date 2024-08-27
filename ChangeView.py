import fiftyone as fo
from fiftyone import ViewField as F
import fiftyone.utils.data.exporters as foude

min_uniqueness = 0 
max_uniqueness = 1 


dataset = fo.load_dataset("06_V7")
#tag_name = "duplicate"

#dataset.delete_labels(tags=tag_name)

if __name__ == "__main__":
        session = fo.launch_app(dataset)
        session.wait()














#label_field = "detections"  # The field that contains annotations
#label_value = "car"            # The specific label you want to filter for

# Create a view that only contains samples with the specified label
'''''
view = (
    dataset
    .match({f"{label_field}.detections.label": label_value})
)
'''
'''
.match_tags(["IR", "Low bitrate"], bool=False)
.match((F("uniqueness") >= min_uniqueness) & (F("uniqueness") <= max_uniqueness))
.sort_by("uniqueness", reverse=True)
'''

#outputPath = "C:/Git/ml/DataManagement/datasets/06_V3 Bootleg/Train_Val_Final_AllTogether"


'''''
foude.export_samples(
    #samples=view,
    export_dir=outputPath,
    dataset_type=fo.types.COCODetectionDataset,
    label_field=label_field,
)
'''


