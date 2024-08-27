import fiftyone as fo
import fiftyone.utils.iou as foui
import fiftyone.core.labels as fol

dataset = fo.load_dataset("06_V7")

det_field_1 = "RT-DETR_101_COCO"  
det_field_2 = "RT-DETR_Bootleg"   
IouDetField = "GTV1_BL1_101COCO"  
className = "car"                

view = (
    dataset
    .filter_labels(
        det_field_1,
        (fo.ViewField("label") == className)
    )
    .filter_labels(
        det_field_2,
        (fo.ViewField("label") == className) 
    )
)

print(f"Number of samples in view: {len(view)}")


count = 0
for sample in view:
    count += 1
    print(count)
    best_detections = []

    detections_1 = sample[det_field_1].detections if sample[det_field_1] else []
    detections_2 = sample[det_field_2].detections if sample[det_field_2] else []

    if (sample[det_field_1].detections != None and sample[det_field_2].detections != None):
        for det1 in detections_1:
            if det1.label == "car":
                for det2 in detections_2:
                    if det2.label == "car":
                        preds = [det1]
                        gts = [det2]
                        iou = foui.compute_ious(gts=gts, preds=preds, classwise=True, tolerance=2)
                        if iou > 0.5: 
                            best_detections.append(det1)
                        else:
                            if det1.confidence > 0.75:
                                best_detections.append(det1)
    elif (sample[det_field_1].detections != None and sample[det_field_2].detections == None):
         for det1 in detections_1:
            if det1.confidence > 0.75:
                best_detections.append(det1)


    sample[IouDetField].detections = best_detections
    sample.save()

dataset.save()
print(dataset)

for sample in view: 
    print(sample[IouDetField])
print("IoU computation and attribute update completed.")



