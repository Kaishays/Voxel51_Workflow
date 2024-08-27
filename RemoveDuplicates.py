import fiftyone as fo
import fiftyone.utils.iou as foui
import fiftyone.core.labels as fcl
dataset = fo.load_dataset("06_V7")

# Define parameters
saved_view_name = "FilterDatasetV1"  
IouDetField = "GTV1_BL1_101COCO"  
iou_thresh = 0.99  
method = "simple"  

view = dataset.load_saved_view(saved_view_name)

duplicate_label_ids = foui.find_duplicates(
    sample_collection=view,
    label_field=IouDetField,
    iou_thresh=iou_thresh,
    method=method,
    progress=True 
)
print(f"Found {len(duplicate_label_ids)} duplicate labels")

tag_name = "duplicate"




dataset.tag_labels(tags=tag_name, label_fields=IouDetField)

print("Review the duplicates in the FiftyOne App and verify the 'duplicate' tags.")

dataset.delete_labels(tags=tag_name)

print(dataset.count_label_tags())
